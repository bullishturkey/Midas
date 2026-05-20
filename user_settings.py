"""
user_settings.py
-----------------
Lightweight JSON-backed store for per-user bot settings.
Each user's settings are keyed by their Discord user ID.

Default settings per user:
    limit_price : float  — limit credit to collect (default $5.00)
    auto_trade  : bool   — whether to auto-execute trades (default True)
"""

import json
import logging
import os
from typing import Any

log = logging.getLogger("midas.settings")

DEFAULT_SETTINGS = {
    "limit_price": 5.00,
    "auto_trade":  True,
}


class UserSettingsStore:
    def __init__(self, filepath: str = "user_settings.json"):
        self.filepath = filepath
        self._data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self._data = json.load(f)
                log.info("Loaded settings from %s", self.filepath)
            except (json.JSONDecodeError, IOError) as e:
                log.warning("Could not load settings file: %s — starting fresh.", e)
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self._data, f, indent=2)
        except IOError as e:
            log.error("Failed to save settings: %s", e)

    def get(self, user_id: str) -> dict:
        """Return full settings dict for a user, with defaults filled in."""
        user_data = self._data.get(user_id, {})
        return {**DEFAULT_SETTINGS, **user_data}

    def set(self, user_id: str, key: str, value: Any):
        """Update a single setting for a user and persist."""
        if user_id not in self._data:
            self._data[user_id] = {}
        self._data[user_id][key] = value
        self._save()
        log.debug("Settings updated — user: %s | %s = %s", user_id, key, value)

    def all_users(self) -> dict:
        """Return all user settings (for admin/reporting use)."""
        return {uid: {**DEFAULT_SETTINGS, **cfg} for uid, cfg in self._data.items()}
