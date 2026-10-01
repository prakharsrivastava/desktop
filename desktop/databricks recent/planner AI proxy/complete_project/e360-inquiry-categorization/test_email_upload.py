"""
test_email_upload.py — .eml file upload test
"""
import requests, os

BASE = os.environ.get("E360_URL", "http://localhost:8001")
CID  = os.environ.get("CLIENT_ID",  "test-client-id")
CSEC = os.environ.get("CLIENT_SECRET", "test-client-secret")

SAMPLE_EML = b"""From: member@example.com
To: support@insurance.com
Subject: Need new ID card urgently

Hello,

I urgently need a new ID card. My current card is lost.
My HCID is XY9876543.

Please send it ASAP.

Thanks
"""

def get_token():
    r = requests.post(f"{BASE}/token", params={"username":"admin","password":"admin123"})
    return r.json()["access_token"]

def test_upload():
    token = get_token()
    r = requests.post(
        f"{BASE}/upload",
        files={"file": ("test.eml", SAMPLE_EML, "message/rfc822")},
        params={"client_id": CID, "client_secret": CSEC},
        headers={"Authorization": f"Bearer {token}"},
    )
    print("Status:", r.status_code)
    import json; print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    print("✅ Upload test passed")

if __name__ == "__main__":
    test_upload()
