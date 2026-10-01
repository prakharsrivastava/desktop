"""
test_credentials.py — Credential aur token test
"""
import requests, os

BASE = os.environ.get("E360_URL", "http://localhost:8001")

def test_login():
    r = requests.post(f"{BASE}/token", params={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "access_token" in data
    print("✅ Login OK — token:", data["access_token"][:30], "...")
    return data["access_token"]

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print("✅ Health OK:", r.json())

if __name__ == "__main__":
    test_health()
    test_login()
