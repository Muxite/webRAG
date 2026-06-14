"""open_meteo_weather action — current conditions from open-meteo.com (free, no key).

API docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

from typing import Any, Dict

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail, fetch_json, ok


class OpenMeteoWeatherAction(LeafAction):
    """Current weather at a lat/lon via Open-Meteo. No API key.

    Reads from node details:
      - `lat` (number): latitude (required).
      - `lon` (number): longitude (required).
      - `units` (str): "metric" (default) or "imperial".

    Returns `{lat, lon, temperature, units, windspeed, weather_code,
    is_day, time}`.
    """

    name = "open_meteo_weather"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        try:
            lat = float(details["lat"])
            lon = float(details["lon"])
        except (KeyError, TypeError, ValueError):
            return fail(self.name, "missing or non-numeric 'lat'/'lon' detail")
        units = (details.get("units") or "metric").lower()
        temp_unit = "fahrenheit" if units == "imperial" else "celsius"
        wind_unit = "mph" if units == "imperial" else "kmh"

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
            f"&temperature_unit={temp_unit}&windspeed_unit={wind_unit}"
        )
        resp = await fetch_json(io, url)
        if not resp.get("_ok"):
            return fail(self.name, resp.get("error", "fetch failed"), retryable=True)
        data = resp["data"]
        current = (data or {}).get("current_weather") or {}
        if not current:
            return fail(self.name, "no current_weather payload returned")
        return ok(
            self.name,
            lat=lat, lon=lon, units=units,
            temperature=current.get("temperature"),
            windspeed=current.get("windspeed"),
            winddirection=current.get("winddirection"),
            weather_code=current.get("weathercode"),
            is_day=bool(current.get("is_day")),
            time=current.get("time"),
        )
