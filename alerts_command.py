"""
alerts_command.py
------------------
All communication between Midas and the Alerts Command app.

Endpoints used:
  GET  /api/midas/subscribers        — fetch all active subscribers
  GET  /api/midas/subscribers/:id    — fetch one subscriber by discord_id
  POST /api/midas/subscribers/:id    — update a subscriber's settings
  POST /api/midas/members            — sync role add/remove
  POST /api/midas/trades             — log a completed trade
"""

import logging
from datetime import datetime, timezone

import aiohttp
import discord

log = logging.getLogger("midas.alerts-command")


class AlertsCommandClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "X-Midas-Key":  self.api_key,
        }

    async def _get(self, path: str):
        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        body = await resp.text()
                        log.warning("GET %s failed: %s — %s", path, resp.status, body)
                        return None
        except Exception as e:
            log.error("GET %s error: %s", path, e)
            return None

    async def _post(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 201):
                        log.info("POST %s OK", path)
                        return await resp.json()
                    else:
                        body = await resp.text()
                        log.warning("POST %s failed: %s — %s", path, resp.status, body)
                        return None
        except Exception as e:
            log.error("POST %s error: %s", path, e)
            return None

    async def get_subscribers(self) -> list:
        """Fetch all active Midas subscribers from Alerts Command."""
        result = await self._get("/midas/subscribers")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("subscribers", result.get("data", []))
        return []

    async def get_subscriber(self, discord_id: str) -> dict:
        """Fetch a single subscriber by Discord ID."""
        result = await self._get(f"/midas/subscribers/{discord_id}")
        if isinstance(result, dict):
            return result.get("subscriber", result.get("data", result))
        return {}

    async def update_subscriber(self, discord_id: str, updates: dict):
        """Update a subscriber's settings (limit_price, auto_trade, etc.)."""
        await self._post(f"/midas/subscribers/{discord_id}", updates)

    async def sync_member(self, member: discord.Member, action: str):
        """
        POST to /midas/members when a user gains or loses the NDX Trader role.
        action: "add" | "remove"
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
        await self._post("/midas/members", payload)

    async def log_trade(
        self,
        discord_id: str,
        display_name: str,
        result: dict,
        ndx_price: float,
    ):
        """POST trade entry data to Alerts Command after every fill."""
        payload = {
            "event":           "trade.executed",
            "discord_id":      discord_id,
            "display_name":    display_name,
            "underlying":      "NDX",
            "price_at_alert":  ndx_price,
            "short_strike":    result.get("short_strike"),
            "long_strike":     result.get("long_strike"),
            "contracts":       result.get("contracts"),
            "limit_price":     result.get("limit_price"),
            "account_balance": result.get("balance"),
            "account_number":  result.get("account"),
            "order_id":        result.get("order_id"),
            "status":          "working" if result.get("success") else "failed",
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        await self._post("/midas/trades", payload)

    async def log_trade_close(
        self,
        discord_id: str,
        display_name: str,
        email: str,
        entry: dict,
        exit_price: float,
        pnl_per_contract: float,
    ):
        """
        POST trade close data to Alerts Command so we can track trial P&L.
        Called when a spread expires worthless (max profit) or is closed early.

        pnl_per_contract: positive = profit (kept credit), negative = loss.
        For a $5 credit spread that expires worthless: pnl = +$5.00 per contract.
        For a max loss: pnl = -(width - credit) per contract.
        """
        contracts = entry.get("contracts", 1)
        total_pnl = round(pnl_per_contract * contracts * 100, 2)  # convert to dollars
        outcome   = "win" if total_pnl > 0 else "loss" if total_pnl < 0 else "scratch"

        payload = {
            "api_key":      self.api_key,
            "discord_id":   discord_id,
            "display_name": display_name,
            "email":        email,
            "pnl":          total_pnl,
            "contracts":    contracts,
            "short_strike": entry.get("short_strike"),
            "long_strike":  entry.get("long_strike"),
            "entry_price":  entry.get("limit_price", 0),
            "exit_price":   exit_price,
            "outcome":      outcome,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }
        log.info("[trade_close] %s | PnL: $%.2f | %s", display_name, total_pnl, outcome)
        await self._post("/midas/trade-close", payload)
