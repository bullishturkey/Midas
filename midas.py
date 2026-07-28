"""
Midas — AI-Powered NDX Trading Bot
------------------------------------
Features:
  - Gemini AI for natural language conversation
  - Public channel: @Midas for market data, news, tickers, sentiment
  - Personal questions auto-redirected to DM
  - DMs: full AI conversation, settings management, account info
  - Live NDX/VIX/SPX data via Yahoo Finance
  - Natural language settings changes (no hard commands needed)
  - Alerts Command integration for subscriber management

Requirements:
    pip install discord.py tastytrade python-dotenv aiohttp groq yfinance

Setup:
    1. Fill in .env
    2. Run: python3 midas.py
"""

import asyncio
import os
import re
import math
import logging
import httpx
from datetime import date, datetime, timezone
from decimal import Decimal

import discord
from discord.ext import commands
from dotenv import load_dotenv
import anthropic
import yfinance as yf

from alerts_command import AlertsCommandClient

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("midas")

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
ALERT_CHANNEL_ID = int(os.getenv("ALERT_CHANNEL_ID", "0"))
BOT_CHANNEL_ID   = int(os.getenv("BOT_CHANNEL_ID", "0"))
TRADER_ROLE_NAME = os.getenv("TRADER_ROLE_NAME", "Midas Trader")
PAPER_TRADING    = os.getenv("PAPER_TRADING", "true").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("Midas_Brain_API_KEY")

UNDERLYING      = "NDX"
ADMIN_DISCORD_ID     = int(os.getenv("ADMIN_DISCORD_ID", "1051881555909427241"))
MILESTONE_CHANNEL_ID = int(os.getenv("MILESTONE_CHANNEL_ID", "1504190249449029692"))
_milestone_tracker: dict = {}  # user_id -> last milestone notified (in thousands)
STRIKE_INTERVAL = 10
SPREAD_WIDTH    = 10
DEFAULT_LIMIT   = 5.00

# ── Anthropic Setup ────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Alerts Command ─────────────────────────────────────────────────────────────
ac = AlertsCommandClient(
    base_url=os.getenv("ALERTS_COMMAND_URL", "https://alert-command-app.onrender.com/api"),
    api_key=os.getenv("ALERTS_COMMAND_KEY"),
)

AC_BASE = os.getenv("ALERTS_COMMAND_URL", "https://alert-command-app.onrender.com/api").rstrip("/api").rstrip("/")
AC_KEY  = os.getenv("ALERTS_COMMAND_KEY", "")

async def push_user(discord_id: str, title: str, message: str, data: dict = None):
    """Send a push notification to a user's iPhone via the Alert Command App backend."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            await c.post(
                f"{AC_BASE}/api/midas/notify-user",
                json={"discord_id": discord_id, "title": title, "message": message, "data": data or {}},
                headers={"X-Midas-Key": AC_KEY},
            )
    except Exception as e:
        log.warning(f"push_user failed for {discord_id}: {e}")


conversation_history: dict = {}

# ── Personal topic keywords ────────────────────────────────────────────────────
PERSONAL_KEYWORDS = [
    "my account", "my balance", "my settings", "my limit", "my trades",
    "my contracts", "my status", "auto trade", "autotrade", "change my",
    "update my", "set my", "turn on", "turn off", "how many contracts",
    "my position", "my profile", "my subscription"
]


# ── Live Market Data ───────────────────────────────────────────────────────────
def get_market_data() -> dict:
    try:
        data = {}
        for label, symbol in [("NDX", "^NDX"), ("VIX", "^VIX"), ("SPX", "^GSPC")]:
            ticker = yf.Ticker(symbol)
            data[label] = round(ticker.fast_info.last_price, 2)
        data["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return data
    except Exception as e:
        log.warning("Market data fetch failed: %s", e)
        return {}


def get_ticker_data(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol.upper())
        info   = ticker.fast_info
        return {
            "symbol": symbol.upper(),
            "price":  round(info.last_price, 2),
            "high":   round(info.day_high, 2),
            "low":    round(info.day_low, 2),
        }
    except Exception as e:
        log.warning("Ticker fetch failed %s: %s", symbol, e)
        return {}


def is_market_open() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=14, minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=21, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


# ── Position Sizing & Strike Logic ────────────────────────────────────────────
def get_contract_count(balance: float) -> int:
    if balance < 7_000:
        return 1
    if balance < 15_000:
        return 2
    return 3 + math.floor((balance - 15_000) / 5_000)


def calculate_strikes(ndx_price: float) -> tuple:
    rounded = int(math.floor(ndx_price / STRIKE_INTERVAL) * STRIKE_INTERVAL)
    short = rounded + STRIKE_INTERVAL
    return short, short - SPREAD_WIDTH


def parse_ndx_price(content: str):
    match = re.search(r"NDX\s*[-–]\s*([\d,]+\.?\d*)", content, re.IGNORECASE)
    return float(match.group(1).replace(",", "")) if match else None


# ── Gemini AI ─────────────────────────────────────────────────────────────────
async def ask_groq(
    user_message: str,
    user_id: str,
    subscriber_info: dict = None,
    market_data: dict = None,
    is_public: bool = False,
    image_data: bytes = None,
    image_media_type: str = None,
) -> str:

    market_str = ""
    if market_data:
        market_str = (
            f"\nLive Market: NDX={market_data.get('NDX','N/A')} | "
            f"SPX={market_data.get('SPX','N/A')} | "
            f"VIX={market_data.get('VIX','N/A')} | "
            f"{market_data.get('timestamp','')}"
        )

    sub_str = ""
    if subscriber_info:
        sub_str = (
            f"\nSubscriber: {subscriber_info.get('display_name','Unknown')} | "
            f"Auto-Trade: {'ON' if subscriber_info.get('auto_trade', True) else 'OFF'} | "
            f"Limit: ${float(subscriber_info.get('limit_price', DEFAULT_LIMIT)):.2f} | "
            f"Tastytrade: {'Connected' if subscriber_info.get('tastytrade_client_secret') else 'Not Connected'}"
        )

    is_admin = subscriber_info.get("is_admin", False) if subscriber_info else False

    system_prompt = f"""You are Midas — the AI market intelligence brain for an elite NDX 0DTE options trading group. You are sharp, data-driven, direct, and deeply knowledgeable.

## Your personality
You are confident but never arrogant. You speak like a seasoned trader who has seen everything — calm under pressure, precise with data, and genuinely helpful. You never hype, never fear-monger. You lay out the facts and respect people's autonomy.

## Your formatting rules (ALWAYS follow these)
- Use **bold** to highlight key numbers, terms, and important phrases
- Organize multi-point responses into clear sections with bold headers like **Market Overview**, **VIX Reading**, **What This Means**
- Use bullet points (•) for lists of 3 or more items — never paragraph walls for lists
- Keep responses focused — lead with the most important point
- For education topics: break into sections (What It Is, How It Works, Key Terms, Example), use bullets, be thorough but scannable
- For market updates: mirror the sentiment in the app (bullish/bearish/neutral) with supporting data
- When you can elaborate on a topic, end with: *"Want me to go deeper on any of this?"*

## What you can do
- Provide real market data, NDX/VIX/SPX levels, sentiment analysis
- Teach options trading concepts in depth — spreads, Greeks, IV, 0DTE, etc.
- Break down trade screenshots or positions if someone shares one
- Analyze market conditions and news
- For admin: look up subscriber accounts, settings, trade history

## What you never do
- Never give direct trade recommendations ("you should buy/sell X")
- Never share subscriber info with non-admins
- Never share account credentials or tokens
- Always end market interpretations with something like: *"That's the data — the call is yours."*
{market_str}{sub_str}

{"**ADMIN MODE** — Full access. You can discuss subscriber accounts, settings, and internal data when asked." if is_admin else "PUBLIC CHANNEL: Be concise. Data, sentiment, news only. No personal account info." if is_public else "PRIVATE DM: Full conversation. Can discuss personal settings. Education mode active."}
"""

    try:
        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        if not is_public:
            # Add conversation history for DMs
            if user_id not in conversation_history:
                conversation_history[user_id] = []
            for msg in conversation_history[user_id][-16:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        if image_data and image_media_type:
            messages.append({"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": image_media_type, "data": __import__("base64").b64encode(image_data).decode()}},
                {"type": "text", "text": user_message or "Please analyze this trade/chart and break it down for me."}
            ]})
        else:
            messages.append({"role": "user", "content": user_message})

        # Strip system message — Anthropic takes it separately
        system_msg = messages[0]['content'] if messages[0]['role'] == 'system' else system_prompt
        user_messages = [m for m in messages if m['role'] != 'system']

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system_msg,
            messages=user_messages,
        )

        reply = response.content[0].text

        if not is_public:
            conversation_history[user_id].append({"role": "user",      "content": user_message})
            conversation_history[user_id].append({"role": "assistant",  "content": reply})

        return reply

    except Exception as e:
        log.error("Anthropic error: %s", e)
        return "I'm having a connection issue right now. Try again in a moment."


# ── Natural Language Settings Parser ──────────────────────────────────────────
def parse_settings_intent(message: str) -> dict | None:
    msg = message.lower()

    limit_match = re.search(
        r"(?:change|set|update|make).*?limit.*?\$?([\d.]+)|\blimit\b.*?\$?([\d.]+)", msg
    )
    if limit_match:
        val = limit_match.group(1) or limit_match.group(2)
        try:
            return {"action": "limit_price", "value": float(val)}
        except:
            pass

    if any(p in msg for p in ["turn on auto", "enable auto", "autotrade on", "auto trade on", "start auto", "auto on"]):
        return {"action": "auto_trade", "value": True}
    if any(p in msg for p in ["turn off auto", "disable auto", "autotrade off", "auto trade off", "stop auto", "pause auto", "auto off"]):
        return {"action": "auto_trade", "value": False}

    return None


def is_personal_message(message: str) -> bool:
    return any(k in message.lower() for k in PERSONAL_KEYWORDS)


# ── Tastytrade Per-User Executor ───────────────────────────────────────────────
class UserTradeExecutor:
    def __init__(self, tt_secret: str, tt_refresh: str):
        self.tt_secret  = tt_secret
        self.tt_refresh = tt_refresh
        self.session    = None
        self.account    = None

    async def connect(self, account_number: str = None):
        from tastytrade import Session
        from tastytrade.account import Account
        self.session = Session(
            provider_secret=self.tt_secret,
            refresh_token=self.tt_refresh,
            is_test=PAPER_TRADING,
        )
        accounts = await Account.get(self.session)
        if isinstance(accounts, list):
            self.account = next(
                (a for a in accounts if a.account_number == account_number),
                accounts[0],
            ) if account_number else accounts[0]
        else:
            self.account = accounts

    async def get_balance(self) -> float:
        balances = await self.account.get_balances(self.session)
        return float(balances.net_liquidating_value)

    async def get_0dte_option(self, strike: int, option_type: str):
        from tastytrade.instruments import Option, NestedOptionChain
        today  = date.today().strftime("%Y-%m-%d")
        chains = await NestedOptionChain.get(self.session, UNDERLYING)
        chain  = chains[0]
        for expiration in chain.expirations:
            if str(expiration.expiration_date) == today:
                for s in expiration.strikes:
                    if int(float(s.strike_price)) == strike:
                        symbol = s.put if option_type == "P" else s.call
                        return await Option.get(self.session, symbol)
        return None

    async def place_spread(self, ndx_price: float, limit_price: float) -> dict:
        from tastytrade.order import (
            NewOrder, OrderAction, OrderTimeInForce, OrderType, Leg, PriceEffect,
        )
        short_strike, long_strike = calculate_strikes(ndx_price)
        balance   = await self.get_balance()
        contracts = get_contract_count(balance)

        short_put = await self.get_0dte_option(short_strike, "P")
        long_put  = await self.get_0dte_option(long_strike, "P")

        if not short_put or not long_put:
            return {"success": False, "error": "Could not fetch 0DTE contracts — market may be closed"}

        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.LIMIT,
            price=Decimal(str(limit_price)),
            price_effect=PriceEffect.CREDIT,
            legs=[
                Leg(instrument_type=short_put.instrument_type, symbol=short_put.symbol,
                    quantity=contracts, action=OrderAction.SELL_TO_OPEN),
                Leg(instrument_type=long_put.instrument_type,  symbol=long_put.symbol,
                    quantity=contracts, action=OrderAction.BUY_TO_OPEN),
            ],
        )

        try:
            response = await self.account.place_order(self.session, order, dry_run=False)
            return {
                "success": True, "short_strike": short_strike, "long_strike": long_strike,
                "contracts": contracts, "limit_price": limit_price,
                "balance": balance, "account": self.account.account_number,
                "order_id": str(response),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_open_option_positions(self) -> list:
        """Return all currently open option positions on this account."""
        from tastytrade.account import Account
        positions = await self.account.get_positions(self.session)
        return [p for p in positions if str(getattr(p, 'instrument_type', '')).lower() in ('equity option', 'option')]

    async def get_spread_net_value(self, short_strike: int, long_strike: int) -> float | None:
        """
        Fetch the current mid-price of the spread (short_put - long_put).
        Returns net debit to close (positive = cost to close). Returns None if unavailable.
        """
        from tastytrade.instruments import Option, NestedOptionChain
        try:
            today = date.today().strftime("%Y-%m-%d")
            chains = await NestedOptionChain.get(self.session, UNDERLYING)
            chain = chains[0]
            short_bid = short_ask = long_bid = long_ask = None
            for expiration in chain.expirations:
                if str(expiration.expiration_date) != today:
                    continue
                for s in expiration.strikes:
                    strike_val = int(float(s.strike_price))
                    if strike_val == short_strike and s.put:
                        opt = await Option.get(self.session, s.put)
                        short_bid = float(getattr(opt, 'bid', 0) or 0)
                        short_ask = float(getattr(opt, 'ask', 0) or 0)
                    if strike_val == long_strike and s.put:
                        opt = await Option.get(self.session, s.put)
                        long_bid = float(getattr(opt, 'bid', 0) or 0)
                        long_ask = float(getattr(opt, 'ask', 0) or 0)
            if None in (short_bid, short_ask, long_bid, long_ask):
                return None
            # Net debit to close = what we pay back: buy short put back, sell long put
            short_mid = (short_bid + short_ask) / 2
            long_mid  = (long_bid + long_ask) / 2
            return round(short_mid - long_mid, 4)  # positive = cost to close
        except Exception:
            return None

    async def close_spread(self, short_strike: int, long_strike: int, contracts: int) -> dict:
        """Buy-to-close the short put and sell-to-close the long put at market."""
        from tastytrade.order import (
            NewOrder, OrderAction, OrderTimeInForce, OrderType, Leg, PriceEffect,
        )
        short_put = await self.get_0dte_option(short_strike, "P")
        long_put  = await self.get_0dte_option(long_strike,  "P")
        if not short_put or not long_put:
            return {"success": False, "error": "Could not locate spread legs to close"}
        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            price_effect=PriceEffect.DEBIT,
            legs=[
                Leg(instrument_type=short_put.instrument_type, symbol=short_put.symbol,
                    quantity=contracts, action=OrderAction.BUY_TO_CLOSE),
                Leg(instrument_type=long_put.instrument_type,  symbol=long_put.symbol,
                    quantity=contracts, action=OrderAction.SELL_TO_CLOSE),
            ],
        )
        try:
            response = await self.account.place_order(self.session, order, dry_run=False)
            return {"success": True, "order_id": str(response)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ── Discord Bot ────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)


def has_trader_role(member) -> bool:
    return any(r.name == TRADER_ROLE_NAME for r in member.roles)


def get_member_from_dm(user):
    for guild in bot.guilds:
        member = guild.get_member(user.id)
        if member:
            return member
    return None


@bot.event
async def on_ready():
    log.info("Midas online as %s", bot.user)
    bot_channel = bot.get_channel(BOT_CHANNEL_ID)
    if bot_channel:
        await bot_channel.send(
            f"🟡 **Midas is online.**\n"
            f"Mode: `{'PAPER' if PAPER_TRADING else 'LIVE'}` | "
            f"Market: `{'OPEN 🟢' if is_market_open() else 'CLOSED 🔴'}`\n"
            f"Tag me with **@Midas** to ask about the market. DM me to manage your account."
        )
    # Start background monitors
    asyncio.create_task(profit_target_monitor())
    asyncio.create_task(eod_close_monitor())


# ── Active trade registry ──────────────────────────────────────────────────────
# Tracks open trades placed this session so monitors can check prices.
# Structure: { discord_id: { short_strike, long_strike, contracts, entry_credit, timestamp } }
_active_trades: dict = {}

def register_active_trade(discord_id: str, short_strike: int, long_strike: int,
                           contracts: int, entry_credit: float):
    """Called after a spread is successfully placed to register it for monitoring."""
    _active_trades[discord_id] = {
        "short_strike": short_strike,
        "long_strike": long_strike,
        "contracts": contracts,
        "entry_credit": entry_credit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def clear_active_trade(discord_id: str):
    _active_trades.pop(discord_id, None)


# ── Profit Target Monitor ──────────────────────────────────────────────────────
async def profit_target_monitor():
    """
    Every 60 seconds during market hours, check each subscriber's open trade.
    If current spread value ≤ (1 - profit_target_pct/100) × entry_credit → close at market.
    e.g. entry credit $5.00, target 50% → close when spread costs ≤ $2.50 to buy back.
    """
    await asyncio.sleep(30)  # brief startup delay
    while True:
        try:
            if is_market_open():
                subscribers = await ac.get_subscribers()
                for sub in subscribers:
                    discord_id = sub.get("discord_id")
                    profit_target_pct = sub.get("profit_target_pct")
                    if not discord_id or not profit_target_pct:
                        continue
                    trade = _active_trades.get(discord_id)
                    if not trade:
                        continue
                    try:
                        executor = UserTradeExecutor(
                            sub.get("tastytrade_client_secret"),
                            sub.get("tastytrade_refresh_token"),
                        )
                        await executor.connect(account_number=sub.get("account_number"))
                        current_value = await executor.get_spread_net_value(
                            trade["short_strike"], trade["long_strike"]
                        )
                        if current_value is None:
                            continue
                        entry = trade["entry_credit"]
                        current_profit_pct = round((entry - current_value) / entry * 100, 1) if entry > 0 else 0
                        target_value = round(entry * (1 - profit_target_pct / 100), 4)

                        # ── Profit milestone push alerts (70%, 80%, 90%) ──────────
                        # Alert at milestone thresholds below the auto-close target
                        # Only fire each milestone once per trade (tracked in trade registry)
                        milestones = trade.get("alerted_milestones", set())
                        for milestone in [50, 60, 70, 80, 90]:
                            if milestone >= profit_target_pct:
                                break  # don't alert below auto-close level
                            if current_profit_pct >= milestone and milestone not in milestones:
                                milestones.add(milestone)
                                _active_trades[discord_id]["alerted_milestones"] = milestones
                                asyncio.create_task(push_user(
                                    discord_id,
                                    title=f"📈 {milestone}% Profit Reached",
                                    message=f"Your NDX {trade['short_strike']}/{trade['long_strike']} spread is at {current_profit_pct:.0f}% profit. Consider closing or holding to your {profit_target_pct}% target.",
                                    data={"type": "profit_alert", "profit_pct": current_profit_pct, "milestone": milestone},
                                ))
                                log.info(f"Profit milestone {milestone}% alert sent to {discord_id}")

                        if current_value <= target_value:
                            result = await executor.close_spread(
                                trade["short_strike"], trade["long_strike"], trade["contracts"]
                            )
                            bot_channel = bot.get_channel(BOT_CHANNEL_ID)
                            mention = f"<@{discord_id}>"
                            if result["success"]:
                                clear_active_trade(discord_id)
                                profit = round((entry - current_value) * trade["contracts"] * 100, 2)
                                log.info(f"Profit target hit for {discord_id} — closed at ${current_value:.2f} (target {profit_target_pct}%) P&L ~${profit}")
                                # Push notification — profit target reached
                                asyncio.create_task(push_user(
                                    discord_id,
                                    title=f"✅ {profit_target_pct}% Profit Target Hit!",
                                    message=f"Your spread was auto-closed. Estimated P&L: +${profit:,.2f}",
                                    data={"type": "profit_target_closed", "profit": profit},
                                ))
                                if bot_channel:
                                    await bot_channel.send(
                                        f"✅ **Profit Target Hit** — {mention}\n"
                                        f"Spread closed at **${current_value:.2f}** "
                                        f"({profit_target_pct}% profit target reached)\n"
                                        f"Estimated P&L: **+${profit:,.2f}**"
                                    )
                            else:
                                log.warning(f"Profit target close failed for {discord_id}: {result.get('error')}")
                                if bot_channel:
                                    await bot_channel.send(
                                        f"⚠️ **Profit Target Close Failed** — {mention}\n"
                                        f"Could not close spread automatically: `{result.get('error')}`\n"
                                        f"Please close manually in Tastytrade."
                                    )
                    except Exception as e:
                        log.warning(f"Profit target check error for {discord_id}: {e}")
        except Exception as e:
            log.warning(f"profit_target_monitor loop error: {e}")
        await asyncio.sleep(60)


# ── EOD Auto-Close Monitor ─────────────────────────────────────────────────────
# Closes any open trade at 12:49 PM PT (= 20:49 UTC) if the spread has NOT reached
# 0.4% gain above entry credit (i.e. the trade is not sufficiently profitable yet).
EOD_CLOSE_HOUR_PT   = 12
EOD_CLOSE_MINUTE_PT = 49
EOD_MIN_GAIN_PCT    = 0.4   # must be at least 0.4% profitable to skip EOD close

_eod_closed_today: set = set()  # discord_ids already closed today

async def eod_close_monitor():
    """
    Polls every 30 seconds. At 12:49 PM PT, closes any open trade where the
    subscriber has eod_close_enabled=True and the position is not ≥0.4% above entry.
    Resets the closed-today registry at midnight PT.
    """
    import pytz
    PT = pytz.timezone("America/Los_Angeles")
    await asyncio.sleep(60)
    last_reset_date = None
    while True:
        try:
            now_pt = datetime.now(PT)
            today = now_pt.date()

            # Reset the daily registry at midnight PT
            if last_reset_date != today:
                _eod_closed_today.clear()
                last_reset_date = today

            # Only act at 12:49 PM PT (±30s window)
            is_eod_window = (
                now_pt.hour == EOD_CLOSE_HOUR_PT
                and now_pt.minute == EOD_CLOSE_MINUTE_PT
            )
            if is_eod_window and is_market_open():
                subscribers = await ac.get_subscribers()
                for sub in subscribers:
                    discord_id = sub.get("discord_id")
                    eod_enabled = sub.get("eod_close_enabled", False)
                    if not discord_id or not eod_enabled:
                        continue
                    if discord_id in _eod_closed_today:
                        continue
                    trade = _active_trades.get(discord_id)
                    if not trade:
                        continue
                    try:
                        executor = UserTradeExecutor(
                            sub.get("tastytrade_client_secret"),
                            sub.get("tastytrade_refresh_token"),
                        )
                        await executor.connect(account_number=sub.get("account_number"))
                        current_value = await executor.get_spread_net_value(
                            trade["short_strike"], trade["long_strike"]
                        )
                        entry = trade["entry_credit"]
                        # Use per-user threshold if set, otherwise global default
                        user_threshold = sub.get("eod_close_threshold_pct", EOD_MIN_GAIN_PCT)
                        if user_threshold is None:
                            user_threshold = EOD_MIN_GAIN_PCT
                        # Determine if trade is above user's threshold
                        # Profit on spread = entry - current_value (what we collected minus what it costs to close)
                        profit_pct = ((entry - current_value) / entry * 100) if (current_value is not None and entry > 0) else 0
                        if profit_pct >= user_threshold:
                            log.info(f"EOD close skipped for {discord_id} — trade is {profit_pct:.2f}% profitable, above {user_threshold}% threshold")
                            continue
                        # Not profitable enough — close it
                        result = await executor.close_spread(
                            trade["short_strike"], trade["long_strike"], trade["contracts"]
                        )
                        bot_channel = bot.get_channel(BOT_CHANNEL_ID)
                        mention = f"<@{discord_id}>"
                        _eod_closed_today.add(discord_id)
                        if result["success"]:
                            clear_active_trade(discord_id)
                            log.info(f"EOD close executed for {discord_id} — {profit_pct:.2f}% profitable at close time")
                            if bot_channel:
                                await bot_channel.send(
                                    f"🕐 **EOD Auto-Close** — {mention}\n"
                                    f"Position closed at **12:49 PM PT** — trade was "
                                    f"{'unprofitable' if profit_pct < 0 else f'only {profit_pct:.2f}% profitable'} "
                                    f"(below your {user_threshold}% threshold).\\n"
                                    f"Close value: **${current_value:.2f}** vs entry **${entry:.2f}**"
                                )
                        else:
                            log.warning(f"EOD close failed for {discord_id}: {result.get('error')}")
                            if bot_channel:
                                await bot_channel.send(
                                    f"⚠️ **EOD Auto-Close Failed** — {mention}\n"
                                    f"Could not auto-close at 12:49 PM PT: `{result.get('error')}`\n"
                                    f"**Please close your position manually in Tastytrade immediately.**"
                                )
                    except Exception as e:
                        log.warning(f"EOD close error for {discord_id}: {e}")
        except Exception as e:
            log.warning(f"eod_close_monitor loop error: {e}")
        await asyncio.sleep(30)


@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    before_roles = {r.name for r in before.roles}
    after_roles  = {r.name for r in after.roles}

    if TRADER_ROLE_NAME in (after_roles - before_roles):
        await ac.sync_member(after, action="add")
        await start_onboarding(after)

    if TRADER_ROLE_NAME in (before_roles - after_roles):
        await ac.sync_member(after, action="remove")


async def start_onboarding(member: discord.Member):
    """
    When a member gains the Midas Trader role, create a private thread
    in #ndx-bot and post the full beautiful onboarding walkthrough.
    Credentials are collected securely via DM only.
    """
    bot_channel = bot.get_channel(BOT_CHANNEL_ID)
    if not bot_channel:
        return

    try:
        # Create a private thread just for this member
        thread = await bot_channel.create_thread(
            name=f"🟡 Midas Setup — {member.display_name}",
            type=discord.ChannelType.private_thread,
            auto_archive_duration=1440,
        )
        await thread.add_user(member)

        # ── Step 0: Welcome embed ─────────────────────────────────────────────
        welcome = discord.Embed(
            title="🟡 Welcome to Midas",
            description=(
                f"Hey {member.mention} — you're in. Midas is your automated NDX trading bot.\n\n"
                "This thread will walk you through connecting your Tastytrade account. "
                "It takes about **5 minutes** and only needs to be done once.\n\n"
                "**Your credentials are encrypted end-to-end.** Midas only receives "
                "permission to place trades — not access your password, personal info, or funds. "
                "You can revoke access from Tastytrade at any time."
            ),
            color=0xFFD700,
        )
        welcome.set_footer(text="Midas • Automated NDX Trading • Powered by Alerts Command")
        await thread.send(embed=welcome)
        await asyncio.sleep(1)

        # ── Step 1 ─────────────────────────────────────────────────────────────
        step1 = discord.Embed(
            title="Step 1 of 4 — Open Tastytrade API Settings",
            description=(
                "Open your browser and go to:\n"
                "```\nmy.tastytrade.com\n```\n"
                "Then navigate to:\n"
                "> **Manage** → **My profile** → **API**\n\n"
                "You'll see two options. Tap **OAuth application**."
            ),
            color=0x1a1a1a,
        )
        step1.add_field(
            name="📱 On Mobile",
            value="Tap the ☰ menu (top left) → scroll to **My profile** → **API**",
            inline=False,
        )
        step1.add_field(
            name="💻 On Desktop",
            value="Click **Manage** in the top nav → expand **My profile** → click **API**",
            inline=False,
        )
        step1.set_footer(text="✅ Tap 'OAuth application' — not 'Manage OAuth grants'")
        await thread.send(embed=step1)
        await asyncio.sleep(1)

        # ── Step 2 ─────────────────────────────────────────────────────────────
        step2 = discord.Embed(
            title="Step 2 of 4 — Create Your OAuth Application",
            description=(
                "Click **Create Application** and fill in the form exactly as shown below."
            ),
            color=0x1a1a1a,
        )
        step2.add_field(name="App Name", value="Anything — e.g. `Midas Bot`", inline=True)
        step2.add_field(name="Callback URL", value="`http://localhost:8000`", inline=True)
        step2.add_field(name="Scopes", value="✅ Check **ALL** available scopes", inline=False)
        step2.add_field(
            name="⚠️ Client Secret — Shown Once",
            value=(
                "After clicking **Create Application**, your **Client Secret** appears.\n"
                "**Copy it immediately** — it is never shown again.\n\n"
                "Save it in your Notes app right now:\n"
                "```\nMidas — Client Secret: [paste here]\n```"
            ),
            inline=False,
        )
        step2.set_footer(text="🔒 Never share your client secret in any chat or email")
        await thread.send(embed=step2)
        await asyncio.sleep(1)

        # ── Step 3 ─────────────────────────────────────────────────────────────
        step3 = discord.Embed(
            title="Step 3 of 4 — Create Your Refresh Token",
            description=(
                "Go back to your OAuth app list. Click **Manage** next to your new app, "
                "then click **Create Grant**.\n\n"
                "A **Refresh Token** will appear — copy it and save it in Notes alongside your Client Secret."
            ),
            color=0x1a1a1a,
        )
        step3.add_field(
            name="📋 Your Notes Should Now Have",
            value=(
                "```\nMidas — Client Secret: xK9mP2...\nMidas — Refresh Token: eyJ0eX...\n```"
            ),
            inline=False,
        )
        step3.add_field(
            name="ℹ️ Good to know",
            value=(
                "The refresh token **never expires** — you only create it once. "
                "Keep it safe. It is the key that lets Midas trade on your account."
            ),
            inline=False,
        )
        step3.set_footer(text="🔒 Store both credentials in Notes — not screenshots or email")
        await thread.send(embed=step3)
        await asyncio.sleep(1)

        # ── Step 4 ─────────────────────────────────────────────────────────────
        step4 = discord.Embed(
            title="Step 4 of 4 — Connect to Alerts Command",
            description=(
                "Open the **Alerts Command** app and go to the **Midas** section.\n\n"
                "Paste your **Client Secret** and **Refresh Token** into the form and tap **Connect Account**.\n\n"
                "Once connected you'll see your Tastytrade account number confirmed. "
                "You can also set your **limit price** (default $5.00) and toggle **auto-trading** on or off."
            ),
            color=0x1a1a1a,
        )
        step4.add_field(
            name="⚙️ Your Default Settings",
            value=(
                "```\nLimit Price:  $5.00\nAuto-Trade:   ON\n```\n"
                "You can change these anytime by DMing me directly."
            ),
            inline=False,
        )
        step4.set_footer(text="Alerts Command → Midas section → Connect Account")
        await thread.send(embed=step4)
        await asyncio.sleep(1)

        # ── Final confirmation ─────────────────────────────────────────────────
        final = discord.Embed(
            title="🟡 You're All Set",
            description=(
                f"{member.mention} once you've connected in Alerts Command, Midas is live.\n\n"
                "**What happens next:**\n"
                "When an alert is posted in <#" + str(ALERT_CHANNEL_ID) + ">, "
                "Midas automatically places your NDX 0DTE put credit spread on your Tastytrade account. "
                "You'll see a confirmation right here in <#" + str(BOT_CHANNEL_ID) + ">.\n\n"
                "**Need to change settings?** Just DM me directly — talk to me naturally:\n"
                "> *\"change my limit to 4.50\"*\n"
                "> *\"turn off auto trading\"*\n"
                "> *\"what's NDX at right now?\"*\n\n"
                "**To revoke access:** Tastytrade → Manage → API → Manage OAuth grants → delete Midas Bot."
            ),
            color=0xFFD700,
        )
        final.add_field(
            name="🔒 Security Reminder",
            value=(
                "Your credentials are encrypted and stored only in Alerts Command. "
                "Never share them in this thread or anywhere else. "
                "Midas only has trading access — nothing more."
            ),
            inline=False,
        )
        final.set_footer(text="Midas • Questions? DM me anytime • Powered by Alerts Command")
        await thread.send(embed=final)

        log.info("Onboarding thread created for %s", member.display_name)

    except Exception as e:
        log.error("Failed to create onboarding thread for %s: %s", member.display_name, e)
        # Fall back to DM if thread creation fails
        try:
            await member.send(
                f"🟡 **Welcome to Midas, {member.display_name}!**\n\n"
                f"Open **Alerts Command → Midas** and enter your Tastytrade OAuth credentials to get started. "
                f"DM me if you need help."
            )
        except:
            pass


async def check_milestone(discord_id: str, display_name: str, balance: float):
    """Send notification when account hits milestone thresholds."""
    milestone_channel = bot.get_channel(MILESTONE_CHANNEL_ID)
    if not milestone_channel:
        return
    milestone_k = int(balance // 5000) * 5  # nearest 5k floor
    if milestone_k < 5:
        return
    last = _milestone_tracker.get(discord_id, 0)
    if milestone_k > last:
        _milestone_tracker[discord_id] = milestone_k
        mention = f"<@{discord_id}>" if discord_id else display_name
        await milestone_channel.send(
            f"🏆 **Account Milestone** — {mention}\n"
            f"**{display_name}** just crossed **${milestone_k:,}** 🎉\n"
            f"Current balance: **${balance:,.2f}**\n"
            f"Keep stacking. 💰"
        )




_last_performance = 0

@bot.command(name="performance")
async def performance(ctx):
    """Post the 2026 NDX performance update as an image."""
    global _last_performance
    import time, aiohttp, io
    if time.time() - _last_performance < 30:
        return
    _last_performance = time.time()
    page_url = "https://bullishturkey.github.io/midas-onboarding/performance.html"
    try:
        screenshot_url = f"https://api.microlink.io/?url={page_url}?v={int(time.time())}&screenshot=true&meta=false&waitFor=2000&force=true&viewport.width=520&element=.embed"
        async with aiohttp.ClientSession() as session:
            async with session.get(screenshot_url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                data = await r.json()
                img_url = data.get("data", {}).get("screenshot", {}).get("url", "")
                if img_url:
                    async with session.get(img_url, timeout=aiohttp.ClientTimeout(total=30)) as img_r:
                        img_bytes = await img_r.read()
                    file = discord.File(io.BytesIO(img_bytes), filename="performance.png")
                    await ctx.send(content="@everyone", file=file)
                else:
                    await ctx.send(f"View performance report: {page_url}")
    except Exception as e:
        await ctx.send(f"Error generating image: {e}\nView at: {page_url}")
    try:
        await ctx.message.delete()
    except Exception:
        pass


@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)
    if message.author == bot.user:
        return

    # ── Alert channel — execute trades ────────────────────────────────────────
    if message.channel.id == ALERT_CHANNEL_ID:
        # Check both message content and embeds for NDX price
        content_to_parse = message.content or ''
        if not content_to_parse and message.embeds:
            for embed in message.embeds:
                parts = []
                if embed.title: parts.append(embed.title)
                if embed.description: parts.append(embed.description)
                for field in embed.fields:
                    parts.append(f"{field.name} {field.value}")
                content_to_parse = chr(10).join(parts)
        ndx_price = parse_ndx_price(content_to_parse)
        if not ndx_price:
            return

        log.info("Alert — NDX: %.2f", ndx_price)
        await message.add_reaction("⚡")

        bot_channel = bot.get_channel(BOT_CHANNEL_ID)
        subscribers = await ac.get_subscribers()

        for sub in subscribers:
            discord_id   = sub.get("discord_id")
            tt_secret    = sub.get("tastytrade_client_secret")
            tt_refresh   = sub.get("tastytrade_refresh_token")
            limit_price  = float(sub.get("limit_price", DEFAULT_LIMIT))
            auto_trade   = sub.get("auto_trade", True)
            account_num  = sub.get("account_number")
            display_name = sub.get("display_name", "Subscriber")

            if not auto_trade or not tt_secret or not tt_refresh:
                continue

            try:
                executor = UserTradeExecutor(tt_secret, tt_refresh)
                await executor.connect(account_number=account_num)
                result = await executor.place_spread(ndx_price, limit_price)
            except Exception as e:
                err_str = str(e)
                # Detect expired/revoked Tastytrade credentials
                if 'invalid_grant' in err_str.lower() or 'invalid jwt' in err_str.lower() or 'invalid_grant' in err_str:
                    # Auto-disable auto_trade so this doesn't fire on every alert
                    try:
                        await ac.update_subscriber(discord_id, {'auto_trade': False})
                    except Exception:
                        pass
                    # DM the user to reconnect
                    if discord_id:
                        try:
                            user_obj = await bot.fetch_user(int(discord_id))
                            await user_obj.send(
                                "⚠️ **Midas Auto-Trader Disabled**\n\n"
                                "Your Tastytrade connection has expired (Invalid JWT / refresh token revoked).\n"
                                "Your auto-trader has been **turned off** to prevent failed trade attempts.\n\n"
                                "**To fix:** Open the Alerts Command app → Midas → disconnect and reconnect your Tastytrade account.\n"
                                "Then re-enable Auto-Trader."
                            )
                        except Exception:
                            pass
                    result = {"success": False, "error": f"Tastytrade credentials expired — auto-trader disabled. Please reconnect in the app. ({err_str})"}
                else:
                    result = {"success": False, "error": err_str}

            mention = f"<@{discord_id}>" if discord_id else display_name

            if result["success"]:
                if bot_channel:
                    await bot_channel.send(
                        f"**Order Working** ⏳ — {mention}\n"
                        f"Midas — NDX 0DTE Put Credit Spread\n"
                        f"Short Put: `{result['short_strike']}` | Long Put: `{result['long_strike']}`\n"
                        f"Contracts: `{result['contracts']}` | Limit: `${result['limit_price']:.2f}`\n"
                        f"Account: `{result['account']}` | Balance: `${result['balance']:,.2f}`\n"
                        f"*Order placed and working — confirmation will post once filled.*"
                    )
                await ac.log_trade(discord_id, display_name, result, ndx_price)
                # Push notification — trade confirmation to iPhone
                if discord_id:
                    asyncio.create_task(push_user(
                        discord_id,
                        title="⚡ Trade Placed — Midas",
                        message=f"NDX {result['short_strike']}/{result['long_strike']} Put Spread · {result['contracts']} contract{'s' if result['contracts'] != 1 else ''} · Limit ${result['limit_price']:.2f}",
                        data={"type": "trade_confirmation", "short_strike": result["short_strike"], "long_strike": result["long_strike"]},
                    ))
                # Register trade for profit target + EOD monitors
                if discord_id:
                    register_active_trade(
                        discord_id,
                        result["short_strike"],
                        result["long_strike"],
                        result["contracts"],
                        float(result["limit_price"]),
                    )
                # Check for milestone
                if discord_id and result.get("balance"):
                    await check_milestone(discord_id, display_name, float(result["balance"]))
            else:
                if bot_channel:
                    await bot_channel.send(f"⚠️ Trade failed for {mention}: `{result.get('error')}`")
        return

    # ── Public channel — @Midas mention ──────────────────────────────────────
    # Only respond if Midas was DIRECTLY mentioned (not via @everyone or @here)
    direct_mention = f"<@{bot.user.id}>" in message.content or f"<@!{bot.user.id}>" in message.content
    if direct_mention and not isinstance(message.channel, discord.DMChannel):
        content = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if not content:
            await message.reply("Ask me anything — market data, options education, NDX levels, news. 🟡")
            return

        # Setup request in public channel → redirect to thread
        if is_setup_request(content):
            await message.reply(
                f"{message.author.mention} — I'm creating your private setup thread in this channel now. "
                f"Check for **🟡 Midas Setup — {message.author.display_name}** and I'll walk you through everything. 🟡"
            )
            await start_onboarding(message.author if isinstance(message.author, discord.Member) else get_member_from_dm(message.author))
            return

        # Personal question → redirect to DM
        if is_personal_message(content):
            await message.reply(
                f"{message.author.mention} — that's personal info. "
                f"I'm sliding into your DMs. 🟡\n"
                f"⚠️ Never share account details or credentials in a public channel."
            )
            try:
                dm = await message.author.create_dm()
                sub         = await ac.get_subscriber(str(message.author.id))
                market_data = get_market_data()
                response    = await ask_groq(content, str(message.author.id), sub, market_data, is_public=False)
                await dm.send(f"🟡 **Midas (Private)**\n\n{response}")
            except discord.Forbidden:
                await message.reply("Couldn't DM you — check your privacy settings.")
            return

        # Public response
        async with message.channel.typing():
            market_data = get_market_data()

            # Ticker lookup
            ticker_match = re.search(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b(?=.*(?:price|quote|trading at))", message.content)
            extra_context = content
            if ticker_match:
                sym         = ticker_match.group(1) or ticker_match.group(2)
                ticker_data = get_ticker_data(sym)
                if ticker_data:
                    extra_context += f"\n[{ticker_data['symbol']}: ${ticker_data['price']} | High: ${ticker_data['high']} | Low: ${ticker_data['low']}]"

            # Check for image attachments in public channel
            pub_image_data = None
            pub_image_type = None
            if message.attachments:
                for att in message.attachments:
                    if any(att.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.get(att.url) as resp:
                                if resp.status == 200:
                                    pub_image_data = await resp.read()
                                    pub_image_type = (att.content_type or "image/jpeg").split(";")[0].strip()
                        break
            response = await ask_groq(extra_context, str(message.author.id), None, market_data, is_public=True, image_data=pub_image_data, image_media_type=pub_image_type)
            chunks   = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for i, chunk in enumerate(chunks):
                prefix = "🟡 " if i == 0 else ""
                await message.reply(f"{prefix}{chunk}")
        return

    # ── DM handler ────────────────────────────────────────────────────────────
    if isinstance(message.channel, discord.DMChannel):
        await handle_dm(message)


SETUP_PHRASES = [
    "take me through the tutorial",
    "walk me through",
    "how do i connect",
    "how do i set you up",
    "how to set up",
    "setup tutorial",
    "set up tutorial",
    "connect my tastytrade",
    "connect tastytrade",
    "link my tastytrade",
    "link tastytrade",
    "how do i get started",
    "getting started",
    "how does this work",
    "how do i use you",
    "onboarding",
    "first time",
    "new here",
    "just joined",
    "what do i do first",
    "where do i start",
    "connect my account",
    "link my account",
    "connect alerts command",
    "how do i connect to alerts",
    "setup midas",
    "set up midas",
    "configure midas",
    "how do i configure",
    "oauth",
    "client secret",
    "refresh token",
    "tastytrade setup",
]

def is_setup_request(message: str) -> bool:
    msg = message.lower()
    return any(phrase in msg for phrase in SETUP_PHRASES)


async def handle_dm(message: discord.Message):
    user    = message.author
    member  = get_member_from_dm(user)
    content = message.content.strip()

    if not content:
        return

    if not member or not has_trader_role(member):
        await user.send(f"⛔ You need the **{TRADER_ROLE_NAME}** role to access Midas.")
        return

    discord_id = str(user.id)

    async with message.channel.typing():
        sub    = await ac.get_subscriber(discord_id)
        intent = parse_settings_intent(content)

        # ── Setup/onboarding request ───────────────────────────────────────────
        if is_setup_request(content):
            await user.send(
                "🟡 Got it — I'm setting up your onboarding thread in the server right now. "
                "Head over to **#ndx-bot** and look for your private setup thread. "
                "I'll walk you through everything step by step."
            )
            await start_onboarding(member)
            return

        # ── Settings change ────────────────────────────────────────────────────
        if intent:
            if intent["action"] == "limit_price":
                val = intent["value"]
                if 0.01 <= val <= 20.00:
                    await ac.update_subscriber(discord_id, {"limit_price": val})
                    await user.send(f"🟡 Done — your limit price is now **${val:.2f}**. I'll use that on your next trade.")
                else:
                    await user.send("That's out of range. Limit price needs to be between $0.01 and $20.00.")
            elif intent["action"] == "auto_trade":
                enabled = intent["value"]
                await ac.update_subscriber(discord_id, {"auto_trade": enabled})
                status = "ON ✅ — I'll fire trades automatically." if enabled else "OFF ❌ — I'll sit out until you turn it back on."
                await user.send(f"🟡 Auto-trading is now **{status}**")
            return

        # ── Admin subscriber lookup ────────────────────────────────────────────
        if str(user.id) == str(ADMIN_DISCORD_ID):
            admin_lookup = re.search(r"(?:look up|find|check|show me|pull up|account for)\s+(.+)", content, re.IGNORECASE)
            if admin_lookup or "subscribers" in content.lower() or "all accounts" in content.lower():
                subscribers = await ac.get_subscribers()
                if "subscribers" in content.lower() or "all accounts" in content.lower():
                    lines = [f"**{s.get('display_name','Unknown')}** — Balance: **${float(s.get('account_balance') or 0):,.2f}** | Auto: {'✅' if s.get('auto_trade') else '❌'} | Discord: `{s.get('discord_id','—')}`" for s in subscribers]
                    summary = f"🟡 **All Subscribers ({len(subscribers)})**\n\n" + "\n".join(lines) if lines else "No subscribers found."
                    await user.send(summary[:1900])
                    return
                elif admin_lookup:
                    query = admin_lookup.group(1).lower().strip()
                    matches = [s for s in subscribers if query in (s.get("display_name","") or "").lower() or query in (s.get("discord_id","") or "")]
                    if matches:
                        s = matches[0]
                        await user.send(
                            f"🟡 **Subscriber: {s.get('display_name','Unknown')}**\n"
                            f"• Discord ID: `{s.get('discord_id','—')}`\n"
                            f"• Balance: **${float(s.get('account_balance') or 0):,.2f}**\n"
                            f"• Account #: `{s.get('account_number','—')}`\n"
                            f"• Auto-Trade: {'✅ ON' if s.get('auto_trade') else '❌ OFF'}\n"
                            f"• Limit Price: **${float(s.get('limit_price', 5)):,.2f}**\n"
                            f"• Midas Enabled: {'✅' if s.get('midas_enabled') else '❌'}"
                        )
                    else:
                        await user.send(f"No subscriber found matching `{query}`.")
                    return

        # ── Image/attachment handling ──────────────────────────────────────────
        image_data = None
        image_media_type = None
        if message.attachments:
            for att in message.attachments:
                if any(att.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]):
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(att.url) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                ct = att.content_type or "image/jpeg"
                                image_media_type = ct.split(";")[0].strip()
                    break

        market_data = get_market_data()
        admin_sub = {"is_admin": True} if str(user.id) == str(ADMIN_DISCORD_ID) else None
        effective_sub = admin_sub if admin_sub else sub
        response    = await ask_groq(content, discord_id, effective_sub, market_data, is_public=False, image_data=image_data, image_media_type=image_media_type)
        chunks      = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for i, chunk in enumerate(chunks):
            prefix = "🟡 " if i == 0 else ""
            await user.send(f"{prefix}{chunk}")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
