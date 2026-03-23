"""Discord webhook notifications for trade events."""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_COLORS = {
    "BUY": 0x22C55E,   # green
    "SELL": 0xEF4444,   # red
    "HOLD": 0x6366F1,   # indigo
    "SL": 0xF59E0B,     # amber
    "TP": 0x3B82F6,     # blue
    "ALERT": 0xEC4899,  # pink
}


async def send_trade_notification(
    action: str,
    symbol: str,
    price: float,
    quantity: float = 0.0,
    pnl_usdt: float | None = None,
    pnl_pct: float | None = None,
    confidence: float = 0.0,
    reasoning: str = "",
    trigger: str | None = None,
) -> None:
    """Send a trade notification to Discord. Silently no-ops if webhook not configured."""
    url = settings.DISCORD_WEBHOOK_URL
    if not url:
        return

    color = _COLORS.get(trigger or action, 0x6366F1)
    title = f"{'🛑 ' if trigger == 'SL' else '🎯 ' if trigger == 'TP' else ''}{action} {symbol}"

    fields = [
        {"name": "Price", "value": f"${price:,.6f}", "inline": True},
        {"name": "Confidence", "value": f"{confidence:.0%}", "inline": True},
    ]
    if quantity > 0:
        fields.append({"name": "Quantity", "value": f"{quantity:.6f}", "inline": True})
    if pnl_usdt is not None:
        sign = "+" if pnl_usdt >= 0 else ""
        fields.append({"name": "P&L", "value": f"{sign}${pnl_usdt:,.2f} ({sign}{pnl_pct or 0:.2f}%)", "inline": True})
    if trigger:
        fields.append({"name": "Trigger", "value": trigger.upper(), "inline": True})

    embed = {
        "title": title,
        "description": reasoning[:200] if reasoning else None,
        "color": color,
        "fields": fields,
        "footer": {"text": f"AI Trader • {settings.MODE.upper()} mode"},
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json={"embeds": [embed]})
            if resp.status_code not in (200, 204):
                logger.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Discord notification failed: %s", exc)


async def send_alert(title: str, message: str) -> None:
    """Send a generic alert (e.g., circuit breaker, errors)."""
    url = settings.DISCORD_WEBHOOK_URL
    if not url:
        return

    embed = {
        "title": f"⚠️ {title}",
        "description": message[:500],
        "color": _COLORS["ALERT"],
        "footer": {"text": f"AI Trader • {settings.MODE.upper()} mode"},
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={"embeds": [embed]})
    except Exception as exc:
        logger.warning("Discord alert failed: %s", exc)
