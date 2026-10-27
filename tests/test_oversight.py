"""Human review of hard adoption calls: hard-call detection, standing rules and the gate."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2.core import oversight, prompts, schedule
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import ChannelSpec
from harness2.demo import DemoAdapter, DemoProposer
from harness2.substrate.codex import CODEX_SURFACE
from harness2.substrate.opencode import OPENCODE_SURFACE

RULES = "- Always accept new guardrails.\n- Never accept deleting a guardrail.\n"


class LowConfidence(DemoProposer):
    def call(self, mission, context_dir, label, **kwargs):
        reply = super().call(mission, context_dir, label, **kwargs)
        if label.startswith("experience."):
            reply.text = reply.text.replace("confidence: high", "confidence: low, one rollout")
        return reply


class Conflicting(DemoProposer):
    def call(self, mission, context_dir, label, **kwargs):
        reply = super().call(mission, context_dir, label, **kwargs)
        if re.match(r"(agg|c)\d+\.a\d+$", label):
            reply.text += "\n<conflict>inv1 and dom1 each helped but rewrite one line</conflict>"
        return reply


class Checker:
    """A stand-in for the tool-free rule-check session."""

    def __init__(self, text="", error=None):
        self.text, self.error, self.missions = text, error, []

    def call(self, mission, context_dir, label, **kwargs):
        self.missions.append((mission, context_dir, label))
        if self.error:
            raise self.error
        return Reply(label, self.text, True, cost_usd=0.5)


class OversightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root)
        self.spec = self.adapter.spec.channels["proposer"]

    def run_stream(self, proposer, review=None, k=1):
        return schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            k,
            adapter=self.adapter,
            prop=proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library",
            review=review,
        )

    def library(self, mode):
        return Library(self.root / "library" / mode, "accounting", self.adapter.substrate.surface())

    def selection(self, task_id, mode, k=1):
        path = self.adapter.spec.runs_root / f"demo-oc-agg{k}" / task_id / "summary.json"
        return json.loads(path.read_text())["selections"][mode]

    def test_off_records_hard_calls_and_keeps_the_published_behaviour(self):
        surface = self.adapter.substrate.surface()
        bp = self.adapter.spec.prompts
        self.assertNotIn("<conflict>", prompts.aggregate(bp, surface, k=2))
        self.assertNotIn("<conflict>", prompts.compose_round(bp, surface, t=1, k=2))
        self.assertIn("<conflict>", prompts.aggregate(bp, surface, k=2, conflicts=True))
        self.assertEqual(
            prompts.aggregate(bp, surface, k=2),
            prompts.aggregate(bp, surface, k=2, conflicts=False),
        )

        self.run_stream(LowConfidence(self.spec))
        for mode in ("agg", "seq"):
            self.assertEqual(self.library(mode).entries(), ["ledger-1", "ledger-2"])
            review = self.selection("ledger-1", mode)["review"]
            self.assertEqual((review["adopt"], review["by"]), (True, "auto"))
            self.assertEqual([c["kind"] for c in review["concerns"]], ["weak_evidence"])

    def test_hold_keeps_hard_calls_out_of_the_library_across_resume(self):
        log = self.root / "reviews.jsonl"
        review = oversight.Policy(mode="hold", log=log)
        proposer = LowConfidence(self.spec)
        self.run_stream(proposer, review)
        for mode in ("agg", "seq"):
            self.assertEqual(self.library(mode).entries(), [])
            gate = self.selection("ledger-1", mode)["review"]
            self.assertEqual((gate["adopt"], gate["by"]), (False, "held"))
        self.assertEqual(len(log.read_text().splitlines()), 4)

        resumed = self.run_stream(proposer, review)
        self.assertEqual([row["status"] for row in resumed["tasks"]], ["skipped", "skipped"])
        self.assertEqual(self.library("agg").entries(), [])

    def test_no_hard_call_is_adopted_without_asking(self):
        ask = Mock(side_effect=AssertionError("nobody should be asked"))
        self.run_stream(DemoProposer(self.spec), oversight.Policy(mode="ask", ask=ask))
        self.assertEqual(self.library("agg").entries(), ["ledger-1", "ledger-2"])
        self.assertEqual(self.selection("ledger-2", "seq")["review"]["by"], "auto")

    def test_the_person_decides_a_compose_conflict(self):
        answers = iter(["y", "n", "yes", "no"])
        asked = []

        def ask(question):
            asked.append(question)
            return next(answers)

        with patch("builtins.print"):
            self.run_stream(Conflicting(self.spec), oversight.Policy(mode="ask", ask=ask))
        self.assertEqual(len(asked), 4)
        decisions = [
            (
                self.selection(task, mode)["review"]["by"],
                self.selection(task, mode)["review"]["adopt"],
            )
            for task in ("ledger-1", "ledger-2")
            for mode in ("agg", "seq")
        ]
        self.assertEqual(
            decisions,
            [("person", True), ("person", False), ("person", True), ("person", False)],
        )
        concern = self.selection("ledger-1", "agg")["review"]["concerns"][0]
        self.assertEqual(concern["kind"], "conflict")
        self.assertIn("agg1: inv1 and dom1", concern["detail"])
        self.assertEqual(self.library("agg").entries(), ["ledger-1", "ledger-2"])
        self.assertEqual(self.library("seq").entries(), [])

    def test_ask_without_a_terminal_holds(self):
        with patch("sys.stdin", Mock(isatty=Mock(return_value=False))):
            self.run_stream(LowConfidence(self.spec), oversight.Policy(mode="ask"))
        self.assertEqual(self.selection("ledger-1", "agg")["review"]["by"], "held")

    def test_guardrail_changes(self):
        before = {
            "guardrails.md": "- Never invent a citation.\n- Cite the page.\n",
            "plugin/block_rm.ts": "export default {}\n",
        }
        after = {"guardrails.md": "- Cite the page.\n- Check totals twice.\n"}
        weakened, added = oversight.guardrail_changes(OPENCODE_SURFACE, before, after)
        self.assertEqual(len(weakened), 2)
        self.assertIn("removes plugin/block_rm.ts", weakened)
        self.assertTrue(weakened[0].startswith("drops 1 line(s) from guardrails.md"))
        self.assertEqual(added, ["adds 1 line(s) to guardrails.md"])
        _w, added_only = oversight.guardrail_changes(OPENCODE_SURFACE, after, before)
        self.assertIn("adds plugin/block_rm.ts", added_only)

        hooks = {".codex/hooks.json": '{"PreToolUse": []}\n', "AGENTS.md": "rules\n"}
        weakened, _a = oversight.guardrail_changes(CODEX_SURFACE, hooks, {"AGENTS.md": "rules\n"})
        self.assertEqual(weakened, ["removes .codex/hooks.json"])
        concerns = oversight.hard_calls(
            CODEX_SURFACE, hooks, {"AGENTS.md": "rules\n"}, experience={"confidence": "medium"}
        )
        self.assertEqual([c.kind for c in concerns], ["guardrail"])

    def review(self, checker, **kw):
        policy = oversight.Policy(mode="hold", rules=RULES, checker=checker)
        args = dict(
            task_id="t",
            mode="agg",
            k=1,
            label="agg1",
            surface=OPENCODE_SURFACE,
            base_tree={"guardrails.md": "- Never invent a citation.\n"},
            chosen_tree={"guardrails.md": "- Never invent a citation.\n- Cite the page.\n"},
            experience={"confidence": "high"},
        )
        args.update(kw)
        return oversight.review_adoption(policy, **args)

    def test_standing_rules_settle_what_they_cover(self):
        approve = Checker("```verdict\nAPPROVE\n```\n<rule>Always accept new guardrails.</rule>")
        gate = self.review(approve, experience={"confidence": "low"})
        self.assertEqual((gate.adopt, gate.by, gate.verdict), (True, "rule", "APPROVE"))
        mission, context_dir, label = approve.missions[0]
        self.assertIsNone(context_dir)
        self.assertEqual(label, "review.agg1")
        self.assertIn("Never accept deleting a guardrail.", mission)
        self.assertIn("<<<ADOPTION", mission)

        reject = Checker("```verdict\nREJECT\n```\n<rule>Never accept deleting a guardrail.</rule>")
        gate = self.review(reject)
        self.assertEqual((gate.adopt, gate.by), (False, "rule"))

        none = Checker("```verdict\nNONE\n```\n<rule></rule>")
        self.assertEqual(self.review(none).by, "auto")
        self.assertEqual(self.review(none, experience={}).by, "held")

    def test_an_unreadable_rule_check_goes_to_the_person(self):
        replies = [
            Checker("I think this is fine."),
            Checker("```verdict\nAPPROVE\n```\n<rule>Always accept anything.</rule>"),
            Checker(error=ProposerError("timeout", attempts=2, cost_usd=1)),
        ]
        for checker in replies:
            with self.subTest(text=checker.text):
                gate = self.review(checker)
                self.assertEqual((gate.adopt, gate.by, gate.verdict), (False, "held", "UNCLEAR"))

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            oversight.Policy(mode="maybe")
        with self.assertRaises(ValueError):
            oversight.Policy(mode="off", rules=RULES, checker=Checker())
        with self.assertRaises(ValueError):
            oversight.Policy(mode="hold", rules=RULES)

    def test_tool_free_channels(self):
        seen = []

        def run(command, **kwargs):
            seen.append((command, kwargs.get("env") or {}))
            return Mock(
                stdout='{"result": "ok", "subtype": "success", "total_cost_usd": 0}',
                stderr="",
                returncode=0,
            )

        with patch("harness2.core.proposer.subprocess.run", side_effect=run):
            Proposer(ChannelSpec("claude_cli", "claude-x", tools=False)).call("m", None, "a")
            Proposer(ChannelSpec("claude_cli", "claude-x")).call("m", None, "b")
        self.assertEqual(seen[0][0][-2:], ["--tools", ""])
        self.assertNotIn("--tools", seen[1][0])

        def run_opencode(command, **kwargs):
            seen.append((command, kwargs.get("env") or {}))
            return Mock(stdout='{"type":"text","part":{"text":"ok"}}', stderr="", returncode=0)

        with (
            patch("harness2.core.proposer._opencode_dir", return_value=self.root),
            patch("harness2.core.proposer.subprocess.run", side_effect=run_opencode),
        ):
            Proposer(ChannelSpec("opencode", "p/m", tools=False)).call("m", None, "c")
        permission = json.loads(seen[2][1]["OPENCODE_PERMISSION"])
        self.assertEqual({permission[t] for t in ("edit", "bash", "webfetch")}, {"deny"})


if __name__ == "__main__":
    unittest.main()
