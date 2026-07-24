import os

import aiohttp

from src.utils.logger import logger


async def send_alert(title: str, message: str, level: str = "error") -> bool:
    """
    Send an asynchronous alert to a webhook (e.g., Slack, Discord, MS Teams).
    Requires WEBHOOK_URL environment variable.
    """
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        logger.debug(f"Alert not sent (no WEBHOOK_URL): [{level.upper()}] {title} - {message}")
        return False

    color_map = {"info": "#3498db", "warning": "#f1c40f", "error": "#e74c3c", "critical": "#992d22"}

    # Generic payload format that works well with Slack/Discord
    payload = {
        "attachments": [
            {
                "color": color_map.get(level.lower(), "#95a5a6"),
                "title": f"🚨 {title}" if level in ["error", "critical"] else title,
                "text": message,
                "footer": "SmartRoute-AI Alert System",
            }
        ]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, timeout=5.0) as resp:
                if resp.status >= 400:
                    logger.error(f"Failed to send alert to webhook. Status: {resp.status}")
                    return False
                return True
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
        return False
