# api_client.py

"""
Small helper to send closed-trade events to the Crypto AI API service.
"""

import os
from typing import Dict, Any

import requests

# Base URL of your Flask API on Render
API_BASE_URL = os.getenv(
    "CRYPTO_AI_API_URL",
    "https://crypto-ai-api-h921.onrender.com",  # default if env var not set
)


def send_training_event_to_api(event: Dict[str, Any]) -> None:
    """
    Send a single closed-trade event to the API.

    `event` is the dict we build in main.py (entry_time, exit_time, pnl_usd, etc).
    """
    url = f"{API_BASE_URL}/training-events"  # <-- important: /training-events

    try:
        resp = requests.post(url, json=event, timeout=10)

        if resp.status_code != 200:
            # Log the first part of the response for debugging
            print(
                f"[API_CLIENT] Error {resp.status_code} sending event: "
                f"{resp.text[:200]}"
            )
        else:
            print("[API_CLIENT] Sent training event to API")

    except Exception as e:
        print(f"[API_CLIENT] Exception sending training event to API: {e}")
