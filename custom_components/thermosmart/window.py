"""Window open/close detection – ThermoSmart WindowMixin."""
from __future__ import annotations

from datetime import timedelta

from homeassistant.util import dt as dt_util


class WindowMixin:
    """Fenstererkennung mit konfigurierbaren Verzögerungen."""

    def _check_window_open(self, cfg: dict) -> bool:
        now = dt_util.now()
        open_delay = timedelta(minutes=cfg.get("window_open_delay", 5))
        close_delay = timedelta(minutes=cfg.get("window_close_delay", 2))

        for ws_id in cfg.get("window_sensors", []):
            if not ws_id:
                continue
            ws = self.hass.states.get(ws_id)
            if ws is None:
                continue

            if ws.state == "on":
                if ws_id not in self._window_open_at:
                    # Beim Start bereits offen → sofort effektiv (Delay bereits abgelaufen)
                    self._window_open_at[ws_id] = now - open_delay
                if now - self._window_open_at[ws_id] >= open_delay:
                    return True
            else:
                # Fenster zu – ggf. noch in Schließ-Toleranzzeit
                if ws_id in self._window_close_at:
                    if now - self._window_close_at[ws_id] < close_delay:
                        return True
                    self._window_close_at.pop(ws_id, None)
                self._window_open_at.pop(ws_id, None)

        return False
