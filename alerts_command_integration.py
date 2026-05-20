"""
alerts_command_integration.py
------------------------------
Handles all communication between the NDX trading bot and Alerts Command.

Two responsibilities:
  1. sync_member  — POST when a user gains/loses the trader role
  2. log_trade    — POST trade execution data after every fill

Alerts Command receives these as webhook events and can:
  - Display a live trade feed in the dashboard
  - Track which members are active subscribers
  - Show per-user trade history and settings
"""

import logging
from datetime import datetime, timezone

import aiohttp
import discord

log = logging.getLogger("midas.alerts-command")


class AlertsCommandClient:
    def __init__(self, webhook_url: str, api_key: str):
        self.webhook_url = webhook_url
        self.api_key     = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Source":      "midas",
        }

    async def _post(self, endpoint: str, payload: dict):
        if not self.webhook_url:
            log.warning("ALERTS_COMMAND_WEBHOOK_URL not set — skipping sync.")
            return

        url = f"{self.webhook_url.rstrip('/')}/{endpoint}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 201):
                        log.info("Alerts Command sync OK [%s]: %s", endpoint, resp.status)
                    else:
                        body = await resp.text()
                        log.warning(
                            "Alerts Command sync failed [%s]: %s — %s",
                            endpoint, resp.status, body,
                        )
        except Exception as e:
            log.error("Alerts Command request error [%s]: %s", endpoint, e)

    async def sync_member(self, member: discord.Member, action: str):
        """
        POST to /members when a user gains or loses the trader role.

        action: "add" | "remove"

        Payload shape:
        {
            "event":        "member.role_change",
            "action":       "add" | "remove",
            "discord_id":   "123456789",
            "username":     "john#1234",
            "display_name": "John",
            "avatar_url":   "https://...",
            "timestamp":    "2026-05-13T09:00:00Z"
        }
        """
        payload = {
            "event":        "member.role_change",
            "action":       action,
            "discord_id":   str(member.id),
            "username":     str(member),
            "display_name": member.display_name,
            "avatar_url":   str(member.display_avatar.url) if member.display_avatar else "",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }
        await self._post("members", payload)

    async def log_trade(self, member: discord.Member, result: dict, ndx_price: float):
        """
        POST trade execution data to /trades after every fill.

        Payload shape:
        {
            "event":           "trade.executed",
            "discord_id":      "123456789",
            "display_name":    "John",
            "underlying":      "NDX",
            "price_at_alert":  28727.36,
            "short_strike":    28720,
            "long_strike":     28710,
            "contracts":       2,
            "limit_price":     5.00,
            "account_balance": 12500.00,
            "order_id":        "tt-order-id",
            "timestamp":       "2026-05-13T09:00:00Z"
        }
        """
        payload = {
            "event":           "trade.executed",
            "discord_id":      str(member.id),
            "display_name":    member.display_name,
            "underlying":      "NDX",
            "price_at_alert":  ndx_price,
            "short_strike":    result.get("short_strike"),
            "long_strike":     result.get("long_strike"),
            "contracts":       result.get("contracts"),
            "limit_price":     result.get("limit_price"),
            "account_balance": result.get("balance"),
            "order_id":        result.get("order_id"),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        await self._post("trades", payload)
