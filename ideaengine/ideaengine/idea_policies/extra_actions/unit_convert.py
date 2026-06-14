"""unit_convert action — convert between common units.

Supports a curated set of length / mass / temperature / time / data units.
Not a full physics unit system; deliberately tiny so it has zero deps and
fast lookup. Add units by extending `_FACTORS` or `_TEMPERATURE_CONVERTERS`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from ideaengine.idea_policies.actions import LeafAction
from ideaengine.idea_policies.extra_actions.base import fail, ok


# All factors are "how many SI base units one unit represents".
# Convert by: si = value * factor[unit]; out = si / factor[target_unit].
_FACTORS: Dict[str, Tuple[str, float]] = {
    # Length (SI: meter)
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "mi": ("length", 1609.344),
    "yd": ("length", 0.9144),
    "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    # Mass (SI: kilogram)
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "mg": ("mass", 1e-6),
    "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "ton": ("mass", 1000.0),
    # Time (SI: second)
    "s": ("time", 1.0),
    "ms": ("time", 0.001),
    "min": ("time", 60.0),
    "hr": ("time", 3600.0),
    "day": ("time", 86400.0),
    "week": ("time", 604800.0),
    # Data (SI-ish: byte; use IEC for storage clarity)
    "B": ("data", 1.0),
    "KB": ("data", 1000.0),
    "MB": ("data", 1_000_000.0),
    "GB": ("data", 1_000_000_000.0),
    "TB": ("data", 1_000_000_000_000.0),
    "KiB": ("data", 1024.0),
    "MiB": ("data", 1024.0 ** 2),
    "GiB": ("data", 1024.0 ** 3),
    "TiB": ("data", 1024.0 ** 4),
}


def _c_to_k(c: float) -> float: return c + 273.15
def _k_to_c(k: float) -> float: return k - 273.15
def _f_to_k(f: float) -> float: return (f - 32) * 5 / 9 + 273.15
def _k_to_f(k: float) -> float: return (k - 273.15) * 9 / 5 + 32


_TEMP_TO_KELVIN: Dict[str, Callable[[float], float]] = {"C": _c_to_k, "F": _f_to_k, "K": lambda k: k}
_TEMP_FROM_KELVIN: Dict[str, Callable[[float], float]] = {"C": _k_to_c, "F": _k_to_f, "K": lambda k: k}


class UnitConvertAction(LeafAction):
    """Convert a numeric value between units of the same category.

    Reads from node details:
      - `value` (number): the input value.
      - `from_unit` (str): source unit (e.g. "ft").
      - `to_unit` (str): target unit (e.g. "m").

    Returns `{value, from_unit, to_unit, result, category}`.
    """

    name = "unit_convert"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        try:
            value = float(details["value"])
        except (KeyError, TypeError, ValueError):
            return fail(self.name, "missing or non-numeric 'value' detail")
        from_unit = details.get("from_unit")
        to_unit = details.get("to_unit")
        if not from_unit or not to_unit:
            return fail(self.name, "missing 'from_unit' or 'to_unit' detail")

        # Temperature is non-linear → handled separately.
        if from_unit in _TEMP_TO_KELVIN and to_unit in _TEMP_FROM_KELVIN:
            kelvin = _TEMP_TO_KELVIN[from_unit](value)
            result = _TEMP_FROM_KELVIN[to_unit](kelvin)
            return ok(self.name, value=value, from_unit=from_unit, to_unit=to_unit,
                      result=result, category="temperature")

        src = _FACTORS.get(from_unit)
        dst = _FACTORS.get(to_unit)
        if not src or not dst:
            return fail(self.name, f"unknown unit (supported: {sorted(_FACTORS)})")
        if src[0] != dst[0]:
            return fail(self.name, f"category mismatch: {from_unit} is {src[0]}, {to_unit} is {dst[0]}")
        si = value * src[1]
        result = si / dst[1]
        return ok(self.name, value=value, from_unit=from_unit, to_unit=to_unit,
                  result=result, category=src[0])
