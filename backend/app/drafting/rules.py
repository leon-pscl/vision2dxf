"""Versioned drafting rule registry.

Every principal dimension and landmark is produced by a registered rule, so each
value can be traced back to an equation, its inputs and a source note.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

REGISTRY_VERSION = "1.0.0"


@dataclass
class DraftingRule:
    id: str
    name: str
    output_name: str
    required_inputs: list[str]
    equation: Callable[[dict[str, float]], float]
    explanation: str
    source_note: str
    valid_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    group: str = "derived"
    auto_inputs: bool = False


@dataclass
class RuleTrace:
    rule_id: str
    name: str
    output_name: str
    group: str
    inputs: dict[str, float]
    value: float
    explanation: str
    source_note: str

    def as_dict(self) -> dict:
        return {
            "ruleId": self.rule_id,
            "name": self.name,
            "outputName": self.output_name,
            "group": self.group,
            "inputs": self.inputs,
            "value": self.value,
            "explanation": self.explanation,
            "sourceNote": self.source_note,
        }


class _Probe(dict):
    """Records every key an equation reads so required_inputs never drifts."""

    def __init__(self) -> None:
        super().__init__()
        self.accessed: list[str] = []

    def __missing__(self, key: str) -> float:  # pragma: no cover - trivial
        self.accessed.append(key)
        return 0.0


class RuleRegistry:
    def __init__(self, rules: list[DraftingRule], external_inputs: tuple[str, ...] = ()) -> None:
        self.version = REGISTRY_VERSION
        self._rules = list(rules)
        self.external_inputs = tuple(external_inputs)
        for rule in self._rules:
            if rule.auto_inputs:
                probe = _Probe()
                rule.equation(probe)
                rule.required_inputs = sorted(set(probe.accessed))
        self._assert_topological()

    def _assert_topological(self) -> None:
        produced: set[str] = set(self.external_inputs)
        for rule in self._rules:
            missing = [i for i in rule.required_inputs if i not in produced]
            if missing:
                raise ValueError(
                    f"rule {rule.id} requires {missing} which are not produced by an earlier rule"
                )
            produced.add(rule.output_name)

    def __iter__(self):
        return iter(self._rules)

    def get(self, rule_id: str) -> DraftingRule:
        return next(r for r in self._rules if r.id == rule_id)

    def evaluate(self, inputs: dict[str, float]) -> tuple[dict[str, float], list[RuleTrace]]:
        values = dict(inputs)
        traces: list[RuleTrace] = []
        for rule in self._rules:
            missing = [i for i in rule.required_inputs if i not in values]
            if missing:
                raise ValueError(f"rule {rule.id}: missing inputs {missing}")
            used = {i: values[i] for i in rule.required_inputs}
            value = float(rule.equation(values))
            values[rule.output_name] = value
            traces.append(
                RuleTrace(
                    rule_id=rule.id,
                    name=rule.name,
                    output_name=rule.output_name,
                    group=rule.group,
                    inputs=used,
                    value=value,
                    explanation=rule.explanation,
                    source_note=rule.source_note,
                )
            )
        return values, traces


def landmark_rule(
    rule_id: str,
    output_name: str,
    fn: Callable[[dict[str, float]], float],
    explanation: str,
    source_note: str = "Explicit construction point; see constants in config/drafting.yaml.",
) -> DraftingRule:
    return DraftingRule(
        id=rule_id,
        name=output_name,
        output_name=output_name,
        required_inputs=[],
        equation=fn,
        explanation=explanation,
        source_note=source_note,
        group="landmark",
        auto_inputs=True,
    )
