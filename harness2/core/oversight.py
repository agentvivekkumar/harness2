# Copyright 2026 The agentvivekkumar/harness2 fork contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Human review of adoption: a person sees only the hard calls.

Every adopted harness passes one gate before it enters the library. Local checks flag the
hard calls -- a Compose stage that could not merge two supported edits, a lesson resting on
weak evidence, an edit that removes or weakens a guardrail. Standing rules the person wrote
once settle what they can, through a cheap tool-free model check; a reply the check cannot
read counts as unsettled. What no rule settles goes to the person. Everything else is adopted
automatically.

    off    no gate, the published behaviour; hard calls are still recorded
    ask    ask on the terminal; with no terminal attached, behave as `hold`
    hold   keep the harness out of the library and log why, for the person to read
"""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import prompts as P
from . import tree as harness_tree
from .proposer import Proposer, ProposerError
from .tree import SurfaceSpec

REVIEW_MODES = ("off", "ask", "hold")
VERDICTS = ("APPROVE", "REJECT", "NONE")

_VERDICT = re.compile(r"```verdict\s*\n?\s*([A-Za-z_]+)\s*\n?\s*```", re.S)
_RULE = re.compile(r"<rule>(.*?)</rule>", re.S)

_ASK_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class Concern:
    """One hard call: `conflict`, `weak_evidence` or `guardrail`."""

    kind: str
    detail: str


@dataclass
class Policy:
    """How adoptions are reviewed: the mode, the standing rules and who checks them."""

    mode: str = "off"
    rules: str = ""
    checker: Proposer | None = None
    log: Path | None = None
    ask: Callable[[str], str] | None = None

    def __post_init__(self) -> None:
        if self.mode not in REVIEW_MODES:
            raise ValueError(f"unknown review mode {self.mode!r}; expected one of {REVIEW_MODES}")
        if self.rules.strip() and self.mode == "off":
            raise ValueError("standing rules need --review ask or --review hold")
        if self.rules.strip() and self.checker is None:
            raise ValueError("standing rules need a checker proposer")

    @property
    def active(self) -> bool:
        return self.mode != "off"


@dataclass
class Review:
    """The gate's decision for one adoption, as recorded on the selection."""

    adopt: bool
    by: str
    concerns: list[dict] = field(default_factory=list)
    verdict: str = ""
    rule: str = ""
    note: str = ""
    usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def guardrail_paths(surface: SurfaceSpec, tree: harness_tree.Tree) -> list[str]:
    """The files in `tree` that constrain the agent: guardrail text, plugins and hooks."""
    plugin = None if surface.plugin_dir is None else surface.plugin_dir + "/"
    return sorted(
        rel
        for rel in tree
        if rel in surface.guardrails
        or rel == surface.hooks_file
        or (plugin is not None and rel.startswith(plugin) and rel.endswith(".ts"))
    )


def _lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


def guardrail_changes(
    surface: SurfaceSpec, before: harness_tree.Tree, after: harness_tree.Tree
) -> tuple[list[str], list[str]]:
    """(weakened, added) guardrail changes, one line per file.

    A reworded line counts as a dropped one: a person is asked rather than trusting the
    rewording.
    """
    weakened: list[str] = []
    added: list[str] = []
    for rel in guardrail_paths(surface, before):
        if rel not in after:
            weakened.append(f"removes {rel}")
            continue
        lost = sorted(_lines(before[rel]) - _lines(after[rel]))
        if lost:
            weakened.append(f"drops {len(lost)} line(s) from {rel}, e.g. {lost[0][:160]!r}")
    for rel in guardrail_paths(surface, after):
        if rel not in before:
            added.append(f"adds {rel}")
            continue
        new = _lines(after[rel]) - _lines(before[rel])
        if new:
            added.append(f"adds {len(new)} line(s) to {rel}")
    return weakened, added


def weak_evidence(experience: dict | None) -> str:
    """Why the lesson behind an adoption is weak, or "" when it states medium or high."""
    confidence = str((experience or {}).get("confidence") or "").strip()
    if not confidence:
        return "the lesson states no confidence"
    if confidence.lower().startswith("low"):
        return f"the lesson rates its own confidence low: {confidence[:200]}"
    return ""


def hard_calls(
    surface: SurfaceSpec,
    base_tree: harness_tree.Tree,
    chosen_tree: harness_tree.Tree,
    *,
    experience: dict | None,
    conflicts: Sequence[str] = (),
) -> list[Concern]:
    """Every reason this adoption needs more than an automatic yes."""
    out = [Concern("conflict", c[:600]) for c in conflicts]
    weak = weak_evidence(experience)
    if weak:
        out.append(Concern("weak_evidence", weak))
    weakened, _added = guardrail_changes(surface, base_tree, chosen_tree)
    out += [Concern("guardrail", w) for w in weakened]
    return out


def brief(
    *,
    task_id: str,
    mode: str,
    label: str,
    surface: SurfaceSpec,
    base_tree: harness_tree.Tree,
    chosen_tree: harness_tree.Tree,
    concerns: Sequence[Concern],
    experience: dict | None,
    why: str,
) -> str:
    """What the rule check and the person read: the change, the hard calls and the lesson."""
    changed = harness_tree.diff_summary(surface, base_tree, chosen_tree)
    weakened, added = guardrail_changes(surface, base_tree, chosen_tree)
    exp = experience or {}
    lines = [
        f"task: {task_id}",
        f"mode: {mode}; adopted harness: {label}",
        f"components added: {', '.join(changed['added']) or 'none'}",
        f"components removed: {', '.join(changed['removed']) or 'none'}",
        f"components modified: {', '.join(changed['modified']) or 'none'}",
        f"guardrails weakened: {'; '.join(weakened) or 'none'}",
        f"guardrails added: {'; '.join(added) or 'none'}",
        "hard calls:",
        *([f"- {c.kind}: {c.detail}" for c in concerns] or ["- none"]),
    ]
    lines += [
        f"lesson {key}: {str(exp[key]).strip()[:600]}"
        for key in ("cue", "lesson", "confidence")
        if exp.get(key)
    ]
    if why.strip():
        lines.append(f"why it was chosen: {why.strip()[:1500]}")
    return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def check_rules(policy: Policy, adoption: str, label: str) -> tuple[str, str, float]:
    """(verdict, rule, usd): APPROVE, REJECT or NONE from the rules, else UNCLEAR.

    UNCLEAR covers a failed session, an unreadable reply and a cited rule that is not in the
    rules: none of them settle anything.
    """
    assert policy.checker is not None
    try:
        reply = policy.checker.call(P.standing_check(policy.rules, adoption), None, label)
    except ProposerError as exc:
        return "UNCLEAR", f"the rule check failed: {exc}"[:300], exc.cost_usd
    m = _VERDICT.search(reply.text)
    verdict = m.group(1).strip().upper() if m else ""
    r = _RULE.search(reply.text)
    rule = r.group(1).strip() if r else ""
    if verdict not in VERDICTS:
        return "UNCLEAR", "the rule check's reply had no readable verdict", reply.cost_usd
    if verdict != "NONE" and (not rule or _norm(rule) not in _norm(policy.rules)):
        return (
            "UNCLEAR",
            f"the rule check cited a rule that is not in the rules: {rule[:200]!r}",
            reply.cost_usd,
        )
    return verdict, rule, reply.cost_usd


def _ask_person(policy: Policy, adoption: str) -> tuple[bool, str, str]:
    """(adopt, by, note) from the person, or a hold when nobody can be asked."""
    ask = policy.ask
    if ask is None:
        if not sys.stdin.isatty():
            return False, "held", "no terminal to ask on; kept out of the library"
        ask = input
    with _ASK_LOCK:
        print(
            "\n"
            + "=" * 72
            + "\nA person decides whether this harness enters the library.\n"
            + adoption
            + "\n"
            + "=" * 72,
            flush=True,
        )
        try:
            answer = ask("Adopt it into the library? [y/N] ")
        except EOFError:
            return False, "held", "no answer; kept out of the library"
    if str(answer).strip().lower() in ("y", "yes"):
        return True, "person", "approved"
    return False, "person", "declined"


def _log(policy: Policy, row: dict) -> None:
    if policy.log is None:
        return
    with _LOG_LOCK:
        policy.log.parent.mkdir(parents=True, exist_ok=True)
        with policy.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


def review_adoption(
    policy: Policy,
    *,
    task_id: str,
    mode: str,
    k: int,
    label: str,
    surface: SurfaceSpec,
    base_tree: harness_tree.Tree,
    chosen_tree: harness_tree.Tree,
    experience: dict | None,
    why: str = "",
    conflicts: Sequence[str] = (),
) -> Review:
    """Decide whether one adopted harness enters the library, asking a person only if needed.

    A rule that forbids the adoption wins; a rule that allows all of it adopts it. Otherwise
    an adoption with no hard call is adopted, and one with a hard call, or whose rule check
    was unclear, goes to the person.
    """
    concerns = hard_calls(
        surface, base_tree, chosen_tree, experience=experience, conflicts=conflicts
    )
    found = [asdict(c) for c in concerns]
    if not policy.active:
        return Review(True, "auto", found)
    adoption = brief(
        task_id=task_id,
        mode=mode,
        label=label,
        surface=surface,
        base_tree=base_tree,
        chosen_tree=chosen_tree,
        concerns=concerns,
        experience=experience,
        why=why,
    )
    verdict = rule = ""
    usd = 0.0
    if policy.rules.strip():
        verdict, rule, usd = check_rules(policy, adoption, f"review.{mode}{k}")
    if verdict in ("APPROVE", "REJECT"):
        out = Review(verdict == "APPROVE", "rule", found, verdict, rule, usd=usd)
    elif not concerns and verdict != "UNCLEAR":
        return Review(True, "auto", found, verdict, rule, usd=usd)
    else:
        adopt, by, note = (
            _ask_person(policy, adoption)
            if policy.mode == "ask"
            else (False, "held", "held for review; kept out of the library")
        )
        out = Review(adopt, by, found, verdict, rule, note, usd)
    _log(policy, {"task_id": task_id, "mode": mode, "k": k, "label": label, **out.to_dict()})
    return out
