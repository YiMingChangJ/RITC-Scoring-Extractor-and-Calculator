#!/usr/bin/env python3
# dp_bonds_screener_example.py
import os, json, sys
import requests
from base64 import b64encode

BASE_URL = os.getenv("DP_BASE_URL", "https://api.factset.com")
ENDPOINT = "/api/v1/debtInstrument/notation/screener/search"

# --- Auth: Digital Portals uses Basic (username = USERNAME-SERIAL, password = API KEY)
def make_session(username: str, api_key: str, timeout: float = 60.0) -> requests.Session:
    s = requests.Session()
    s.auth = (username, api_key)
    s.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    s.timeout = timeout
    return s

# --- Your filter, verbatim from the example ---
FILTER_BODY = {
    "data": {
        "validation": {
            "onlyActive": True,
            "prices": {
                "latest": {
                    "availableOnly": True,
                    "minimumDate": "2020-09-14"
                }
            }
        },
        "lifeCycle": {
            "maturity": {
                "restriction": {
                    "remainingTermYears": {
                        "minimum": {"value": 5, "inclusive": True},
                        "maximum": {"value": 15, "inclusive": True}
                    }
                }
            },
            "callable": False
        },
        "issuer": {
            "country": {
                # DE=57, FR=87, US=244  (see /basic/region/country/list for the full list)
                "ids": [57, 87, 244]
            }
        },
        "coupon": {
            "occurrence": {
                "type": ["repeated"],
                "frequency": {
                    # Annual = 2 (see /basic/region/frequency/type/list)
                    "ids": [2]
                }
            },
            "currentInterestRate": {
                # Fixed = 2 (see /instrument/coupon/interestRate/type/list)
                "type": {"ids": [2]},
                "value": {"minimum": {"value": 0.10, "inclusive": True}}
            }
        }
    }
}

def run(username: str, api_key: str, limit: int = 50, offset: int = 0):
    sess = make_session(username, api_key)
    url = f"{BASE_URL}{ENDPOINT}"

    # You can add pagination via query params (limit/offset) if your tenant supports it:
    params = {"limit": limit, "offset": offset}

    r = sess.post(url, params=params, data=json.dumps(FILTER_BODY))
    if not (200 <= r.status_code < 300):
        # Attempt to surface DP error payload
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"DP error {r.status_code}: {err}")

    payload = r.json()
    data = payload.get("data", [])
    print(f"Found {len(data)} result(s). Showing instrument.id | instrument.name | instrument.shortName\n")
    for item in data:
        inst = item.get("instrument", {}) or {}
        _id  = inst.get("id")
        name = inst.get("name")
        sname = inst.get("shortName")
        print(f"{_id} | {name} | {sname}")

if __name__ == "__main__":
    user = os.getenv("FACTSET_USER") or os.getenv("DP_USER") or "ROTMAN-2183292SERIAL"
    key  = os.getenv("FACTSET_KEY")  or os.getenv("DP_KEY")  or "7ifZUKo2XxN4AOOtGMlnJp8wVNaz8QMgoMM9ipnb"
    if user.startswith("YOUR-") or key.startswith("YOUR-"):
        print("Set FACTSET_USER/FACTSET_KEY (or DP_USER/DP_KEY) env vars with your credentials.")
        sys.exit(2)
    try:
        run(user, key)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
