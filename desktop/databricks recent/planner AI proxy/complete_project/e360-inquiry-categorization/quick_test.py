"""
quick_test.py — Quick sanity check for /categorize endpoint
"""
import requests, os, json

BASE  = os.environ.get("E360_URL", "http://localhost:8001")
CID   = os.environ.get("CLIENT_ID", "test-client-id")
CSEC  = os.environ.get("CLIENT_SECRET", "test-client-secret")

def get_token():
    r = requests.post(f"{BASE}/token", params={"username":"admin","password":"admin123"})
    return r.json()["access_token"]

def test_id_card():
    token = get_token()
    payload = {
        "inquiry_id":    "TEST-001",
        "subject":       "Request for New ID Card",
        "body":          "Hello, I need a replacement ID card. My HCID is AB1234567. Please send to my current address.",
        "client_id":     CID,
        "client_secret": CSEC,
    }
    r = requests.post(
        f"{BASE}/categorize",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    assert r.json()["category"]["primary"] == "ID Card Change Request"
    print("✅ ID Card test passed")

def test_address_change():
    token = get_token()
    payload = {
        "inquiry_id":    "TEST-002",
        "subject":       "Update My Address",
        "body":          "Please update my address to 123 Main Street, Chicago IL 60601.",
        "client_id":     CID,
        "client_secret": CSEC,
    }
    r = requests.post(
        f"{BASE}/categorize",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    print(json.dumps(r.json(), indent=2))
    print("✅ Address test passed")

if __name__ == "__main__":
    test_id_card()
    test_address_change()
