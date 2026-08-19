import dataclasses
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple


class Availability(IntEnum):
    RUNTIME = 0
    POSTHOC = 1
    ORACLE = 2


class OracleLeak(Exception):
    pass


class LabelFromRuntime(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Signal:
    value: Any
    availability: Availability
    provenance: str

    def combine(self, *others: "Signal") -> "Signal":
        max_avail = max([self.availability] + [o.availability for o in others])
        combined_provenance = " + ".join([self.provenance] + [o.provenance for o in others])
        return Signal(
            value=self.value,
            availability=max_avail,
            provenance=combined_provenance,
        )


class Label:
    def __init__(self, value: Any, derived_from: str):
        if not derived_from:
            raise ValueError("derived_from must not be empty")
        if derived_from.startswith("runtime."):
            raise LabelFromRuntime("A Label cannot be derived from runtime provenance")
        self._value = value
        self._derived_from = derived_from

    def __str__(self) -> str:
        raise OracleLeak("Cannot convert Label to str (OracleLeak)")

    def __format__(self, format_spec: str) -> str:
        raise OracleLeak("Cannot format Label (OracleLeak)")

    def __eq__(self, other: object) -> bool:
        raise OracleLeak("Cannot compare Label for equality (OracleLeak)")

    def __contains__(self, item: object) -> bool:
        raise OracleLeak("Cannot check containment on Label (OracleLeak)")

    def __repr__(self) -> str:
        return f"<Label derived_from={self._derived_from!r}>"

    def expose(self, reason: str) -> Signal:
        if not reason:
            raise ValueError("reason must not be empty")
        return Signal(
            value=self._value,
            availability=Availability.ORACLE,
            provenance=f"label({self._derived_from}): {reason}",
        )


@dataclasses.dataclass(frozen=True)
class PromptContext:
    family: str
    variant: str
    model: str


@dataclasses.dataclass
class Item:
    item_id: str
    cluster: str
    runtime: Dict[str, Any]
    posthoc: Dict[str, Any]
    label: Label


def false_positive_rate(
    n_false_positive: int, n_negative: int, n_positive: int
) -> Tuple[Optional[float], str]:
    if n_negative == 0:
        return None, "No negative-control population available (n_negative is 0)"
    if n_negative < 0.5 * n_positive:
        return (
            None,
            f"Negative-control population too small ({n_negative} < 0.5 * {n_positive})",
        )
    return float(n_false_positive) / float(n_negative), ""
