import requests

# 🔧 CHANGE THIS if your API URL ever changes
API_BASE = "https://crypto-ai-api-h921.onrender.com"
TRADE_ENDPOINT = f"{API_BASE}/trade"


def send_training_event_to_api(event: dict):
    """
    Send one completed trade to the Flask API.
    `event` must contain all the fields the API expects.
    """
    try:
        resp = requests.post(TRADE_ENDPOINT, json=event, timeout=3)
        if resp.status_code >= 400:
            print(f"[API] Error {resp.status_code}: {resp.text}")
        else:
            print(f"[API] Trade sent. Summary: {resp.json().get('summary')}")
    except Exception as e:
        print(f"[API] Failed to send trade to API: {e}")
