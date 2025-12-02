#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactSet Prices API – minimal, robust client
Docs base URL: https://api.factset.com/content

Auth: Basic (username = your FactSet USERNAME-SERIAL, password = your API KEY)
Rate limit: 25 rps (be courteous; this client doesn't spam-concurrent requests)

Endpoints covered:
- /factset-prices/v1/prices                (GET/POST)
- /factset-prices/v1/fixed-income          (GET/POST)
- /factset-prices/v1/references            (GET/POST)
- /factset-prices/v1/returns               (GET/POST)
- /factset-prices/v1/returns-snapshot      (GET/POST)
- /factset-prices/v1/dividends             (GET/POST)
- /factset-prices/v1/splits                (GET/POST)
- /factset-prices/v1/shares                (GET/POST)
- /factset-prices/v1/market-value          (GET/POST)
- /factset-prices/v1/high-low              (GET/POST)
- /batch/v1/status                         (GET/POST)
- /batch/v1/result                         (GET/POST)
"""
from __future__ import annotations

import time
import json
from typing import Iterable, List, Optional, Dict, Any, Union
from dataclasses import dataclass

import requests

try:
    import pandas as pd  # optional
except Exception:  # pragma: no cover
    pd = None  # DataFrame helpers disabled if pandas not installed


DEFAULT_BASE_URL = "https://api.factset.com/content"


class FactSetAPIError(Exception):
    """Raised for non-2xx responses with parsed FactSet error payload when possible."""

@dataclass
class BatchHandle:
    """Simple handle for batch jobs."""
    id: str


def _ids_param(ids: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(ids, str):
        return [ids]
    return list(ids)


class FactSetPricesClient:
    def __init__(
        self,
        username: str,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sess = session or requests.Session()
        self.sess.auth = (username, api_key)
        self.sess.headers.update({
            "Accept": "application/json",
            # When sending a JSON body we'll also set Content-Type explicitly.
        })

    # ------------- Core HTTP helpers -------------
    def _handle_response(self, r: requests.Response) -> Dict[str, Any]:
        if 200 <= r.status_code < 300:
            # For 201 in batch status, still return JSON
            if r.content:
                try:
                    return r.json()
                except json.JSONDecodeError:
                    # Some 201 responses may be empty; fall through with empty dict
                    return {}
            return {}
        # Attempt to parse structured error
        try:
            payload = r.json()
        except Exception:
            payload = {"message": r.text}
        msg = payload.get("message") or payload.get("status") or f"HTTP {r.status_code}"
        raise FactSetAPIError(f"{msg} (status={r.status_code}, url={r.url})")

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self.sess.get(url, params=params, timeout=self.timeout)
        return self._handle_response(r)

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        r = self.sess.post(url, data=json.dumps(body), headers=headers, timeout=self.timeout)
        return self._handle_response(r)

    # ------------- Convenience: DataFrame wrapping -------------
    @staticmethod
    def _maybe_df(payload: Dict[str, Any], *, as_df: bool):
        if not as_df:
            return payload
        if pd is None:
            raise ImportError("pandas is not installed. Run `pip install pandas` or set as_df=False.")
        data = payload.get("data", [])
        return pd.DataFrame(data)

    # ------------- Prices (Equities & Funds) -------------
    def prices_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,    # D, W, M, AM, CQ, FQ, AY, CY, FY
        calendar: Optional[str] = None,     # FIVEDAY, SEVENDAY, LOCAL
        currency: Optional[str] = None,     # e.g., USD; default Local
        adjust: Optional[str] = None,       # SPLIT (default), SPINOFF, DIVADJ, UNSPLIT
        batch: Optional[str] = None,        # 'Y' or 'N' (default)
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: params["startDate"] = start_date
        if end_date:   params["endDate"]   = end_date
        if frequency:  params["frequency"] = frequency
        if calendar:   params["calendar"]  = calendar
        if currency:   params["currency"]  = currency
        if adjust:     params["adjust"]    = adjust
        if batch:      params["batch"]     = batch
        out = self._get("/factset-prices/v1/prices", params)
        return self._maybe_df(out, as_df=as_df)

    def prices_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        adjust: Optional[str] = None,
        batch: Optional[str] = None,  # set 'Y' to request async batch
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: body["startDate"] = start_date
        if end_date:   body["endDate"]   = end_date
        if frequency:  body["frequency"] = frequency
        if calendar:   body["calendar"]  = calendar
        if currency:   body["currency"]  = currency
        if adjust:     body["adjust"]    = adjust
        if batch:      body["batch"]     = batch
        out = self._post("/factset-prices/v1/prices", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Returns -------------
    def returns_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        dividend_adjust: Optional[str] = None,  # PRICE, EXDATE, PAYDATE, EXDATE_C, PAYDATE_C
        rolling_period: Optional[str] = None,   # 1D,1W,1M,3M,6M,52W,2Y,3Y,5Y,10Y
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date:       params["startDate"]     = start_date
        if end_date:         params["endDate"]       = end_date
        if frequency:        params["frequency"]     = frequency
        if calendar:         params["calendar"]      = calendar
        if currency:         params["currency"]      = currency
        if dividend_adjust:  params["dividendAdjust"]= dividend_adjust
        if rolling_period:   params["rollingPeriod"] = rolling_period
        out = self._get("/factset-prices/v1/returns", params)
        return self._maybe_df(out, as_df=as_df)

    def returns_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        dividend_adjust: Optional[str] = None,
        rolling_period: Optional[str] = None,
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date:       body["startDate"]      = start_date
        if end_date:         body["endDate"]        = end_date
        if frequency:        body["frequency"]      = frequency
        if calendar:         body["calendar"]       = calendar
        if currency:         body["currency"]       = currency
        if dividend_adjust:  body["dividendAdjust"] = dividend_adjust
        if rolling_period:   body["rollingPeriod"]  = rolling_period
        out = self._post("/factset-prices/v1/returns", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Returns Snapshot -------------
    def returns_snapshot_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        date: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        dividend_adjust: Optional[str] = None,  # PRICE, EXDATE, EXDATE_C
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if date:            params["date"]           = date
        if calendar:        params["calendar"]       = calendar
        if currency:        params["currency"]       = currency
        if dividend_adjust: params["dividendAdjust"] = dividend_adjust
        out = self._get("/factset-prices/v1/returns-snapshot", params)
        return self._maybe_df(out, as_df=as_df)

    def returns_snapshot_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        date: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        dividend_adjust: Optional[str] = None,
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if date:            body["date"]            = date
        if calendar:        body["calendar"]        = calendar
        if currency:        body["currency"]        = currency
        if dividend_adjust: body["dividendAdjust"]  = dividend_adjust
        out = self._post("/factset-prices/v1/returns-snapshot", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Dividends -------------
    def dividends_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        currency: Optional[str] = None,
        adjust: Optional[str] = None,  # SPLIT, SPINOFF, DIVADJ, UNSPLIT
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: params["startDate"] = start_date
        if end_date:   params["endDate"]   = end_date
        if currency:   params["currency"]  = currency
        if adjust:     params["adjust"]    = adjust
        out = self._get("/factset-prices/v1/dividends", params)
        return self._maybe_df(out, as_df=as_df)

    def dividends_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        currency: Optional[str] = None,
        adjust: Optional[str] = None,
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: body["startDate"] = start_date
        if end_date:   body["endDate"]   = end_date
        if currency:   body["currency"]  = currency
        if adjust:     body["adjust"]    = adjust
        out = self._post("/factset-prices/v1/dividends", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Splits -------------
    def splits_get(self, ids: Union[str, Iterable[str]], *, as_df: bool = True):
        params = {"ids": _ids_param(ids)}
        out = self._get("/factset-prices/v1/splits", params)
        return self._maybe_df(out, as_df=as_df)

    def splits_post(self, ids: Union[str, Iterable[str]], *, as_df: bool = True):
        body = {"ids": _ids_param(ids)}
        out = self._post("/factset-prices/v1/splits", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Shares -------------
    def shares_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        calendar: Optional[str] = None,
        split_adjust: Optional[str] = None,   # SPLIT, UNSPLIT
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date:    params["startDate"]   = start_date
        if end_date:      params["endDate"]     = end_date
        if frequency:     params["frequency"]   = frequency
        if calendar:      params["calendar"]    = calendar
        if split_adjust:  params["splitAdjust"] = split_adjust
        out = self._get("/factset-prices/v1/shares", params)
        return self._maybe_df(out, as_df=as_df)

    def shares_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        calendar: Optional[str] = None,
        split_adjust: Optional[str] = None,
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date:    body["startDate"]   = start_date
        if end_date:      body["endDate"]     = end_date
        if frequency:     body["frequency"]   = frequency
        if calendar:      body["calendar"]    = calendar
        if split_adjust:  body["splitAdjust"] = split_adjust
        out = self._post("/factset-prices/v1/shares", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Market Value -------------
    def market_value_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: params["startDate"] = start_date
        if end_date:   params["endDate"]   = end_date
        if frequency:  params["frequency"] = frequency
        if calendar:   params["calendar"]  = calendar
        if currency:   params["currency"]  = currency
        out = self._get("/factset-prices/v1/market-value", params)
        return self._maybe_df(out, as_df=as_df)

    def market_value_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: body["startDate"] = start_date
        if end_date:   body["endDate"]   = end_date
        if frequency:  body["frequency"] = frequency
        if calendar:   body["calendar"]  = calendar
        if currency:   body["currency"]  = currency
        out = self._post("/factset-prices/v1/market-value", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- High / Low -------------
    def high_low_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        date: Optional[str] = None,
        period: Optional[str] = None,       # 1D,1W,1M,3M,6M,YTD,52W,2Y,3Y,5Y,10Y
        price_type: Optional[str] = None,   # INTRADAY or CLOSE
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        adjust: Optional[str] = None,
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if date:        params["date"]      = date
        if period:      params["period"]    = period
        if price_type:  params["priceType"] = price_type
        if calendar:    params["calendar"]  = calendar
        if currency:    params["currency"]  = currency
        if adjust:      params["adjust"]    = adjust
        out = self._get("/factset-prices/v1/high-low", params)
        return self._maybe_df(out, as_df=as_df)

    def high_low_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        date: Optional[str] = None,
        period: Optional[str] = None,
        price_type: Optional[str] = None,
        calendar: Optional[str] = None,
        currency: Optional[str] = None,
        adjust: Optional[str] = None,
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if date:        body["date"]      = date
        if period:      body["period"]    = period
        if price_type:  body["priceType"] = price_type
        if calendar:    body["calendar"]  = calendar
        if currency:    body["currency"]  = currency
        if adjust:      body["adjust"]    = adjust
        out = self._post("/factset-prices/v1/high-low", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- References -------------
    def references_get(self, ids: Union[str, Iterable[str]], *, as_df: bool = True):
        params = {"ids": _ids_param(ids)}
        out = self._get("/factset-prices/v1/references", params)
        return self._maybe_df(out, as_df=as_df)

    def references_post(self, ids: Union[str, Iterable[str]], *, as_df: bool = True):
        body = {"ids": _ids_param(ids)}
        out = self._post("/factset-prices/v1/references", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Database Rollover -------------
    def database_rollover(self, *, as_df: bool = True):
        out = self._get("/factset-prices/v1/database-rollover", {})
        return self._maybe_df(out, as_df=as_df)

    # ------------- Fixed Income Prices -------------
    def fixed_income_get(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,  # D,M,AM,MTD,CQ,CQTD,AY,CY,CYTD
        as_df: bool = True,
    ):
        params: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: params["startDate"] = start_date
        if end_date:   params["endDate"]   = end_date
        if frequency:  params["frequency"] = frequency
        out = self._get("/factset-prices/v1/fixed-income", params)
        return self._maybe_df(out, as_df=as_df)

    def fixed_income_post(
        self,
        ids: Union[str, Iterable[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        as_df: bool = True,
    ):
        body: Dict[str, Any] = {"ids": _ids_param(ids)}
        if start_date: body["startDate"] = start_date
        if end_date:   body["endDate"]   = end_date
        if frequency:  body["frequency"] = frequency
        out = self._post("/factset-prices/v1/fixed-income", body)
        return self._maybe_df(out, as_df=as_df)

    # ------------- Batch helpers -------------
    def batch_status_get(self, batch_id: str) -> Dict[str, Any]:
        return self._get("/batch/v1/status", {"id": batch_id})

    def batch_status_post(self, batch_id: str) -> Dict[str, Any]:
        return self._post("/batch/v1/status", {"id": batch_id})

    def batch_result_get(self, batch_id: str) -> Dict[str, Any]:
        return self._get("/batch/v1/result", {"id": batch_id})

    def batch_result_post(self, batch_id: str) -> Dict[str, Any]:
        return self._post("/batch/v1/result", {"id": batch_id})

    def poll_batch_until_done(
        self,
        batch_id: str,
        *,
        interval_sec: float = 2.0,
        timeout_sec: float = 600.0,
        use_post: bool = False,
    ) -> Dict[str, Any]:
        """Poll batch status until DONE/FAILED, then fetch result if DONE."""
        t0 = time.time()
        status_func = self.batch_status_post if use_post else self.batch_status_get
        while True:
            status = status_func(batch_id)
            data = status.get("data") or {}
            state = (data.get("status") or "").upper()
            if state == "DONE":
                # Retrieve result
                return (self.batch_result_post(batch_id) if use_post else self.batch_result_get(batch_id))
            if state == "FAILED":
                raise FactSetAPIError(f"Batch {batch_id} FAILED: {data.get('error')}")
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"Timed out waiting for batch {batch_id} to complete.")
            time.sleep(interval_sec)


# -------------------------- Quickstart examples --------------------------
if __name__ == "__main__":
    """
    Usage:
      export FACTSET_USER="USERNAME-SERIAL"
      export FACTSET_KEY="API_KEY"
      python factset_prices_client.py
    """
    import os

    user = os.getenv("ROTMAN-2183292") or "ROTMAN-2183292"
    key = os.getenv("7ifZUKo2XxN4AOOtGMlnJp8wVNaz8QMgoMM9ipnb") or "7ifZUKo2XxN4AOOtGMlnJp8wVNaz8QMgoMM9ipnb"

    client = FactSetPricesClient(user, key)

    # 1) Basic: last close OHLCV for AAPL-US
    try:
        df = client.prices_get(["AAPL-US"])
        print("Prices sample:")
        print(df.head() if isinstance(df, pd.DataFrame) else df)
    except Exception as e:
        print("Prices error:", e)

    # 2) One month of daily prices for AAPL (2019-03)
    try:
        df = client.prices_post(
            ["AAPL-US"],
            start_date="2019-03-01",
            end_date="2019-03-31",
            frequency="D",
            calendar="FIVEDAY",
            adjust="SPLIT",
        )
        print("\nMar 2019 daily OHLCV (AAPL):", df.shape if isinstance(df, pd.DataFrame) else "ok")
    except Exception as e:
        print("Historical prices error:", e)

    # 3) Snapshot returns for multiple tickers (PRICE change; no reinvest)
    try:
        df = client.returns_snapshot_get(["AAPL-US", "MSFT-US"], dividend_adjust="PRICE")
        print("\nReturns snapshot columns:")
        if isinstance(df, pd.DataFrame):
            print(df.columns.tolist())
        else:
            print("ok")
    except Exception as e:
        print("Returns snapshot error:", e)

    # 4) Example: Rolling 52W total return (compound on ex-date)
    try:
        df = client.returns_get(
            ["AAPL-US"],
            frequency="D",
            dividend_adjust="EXDATE_C",
            rolling_period="52W"
        )
        print("\nRolling 52W return rows:", len(df) if isinstance(df, pd.DataFrame) else "ok")
    except Exception as e:
        print("Returns error:", e)

    # 5) High/Low over the last 52W
    try:
        df = client.high_low_get(["GOOGL-US"], period="52W", price_type="CLOSE")
        print("\nHigh/Low sample:")
        print(df.head() if isinstance(df, pd.DataFrame) else df)
    except Exception as e:
        print("High/Low error:", e)

    # 6) Batch example (submit with batch='Y' via prices_post)
    # NOTE: You must have Batch access enabled on your FactSet account.
    try:
        resp = client.prices_post(
            ["AAPL-US", "MSFT-US"], start_date="2019-01-01", end_date="2019-12-31",
            frequency="D", calendar="FIVEDAY", batch="Y", as_df=False
        )
        # A 202 Accepted response includes a data.id in /batch/v1/status responses;
        # here, we fetch it from the Location header via status polling. The POST
        # response body itself may be minimal; use /batch/v1/status polling below:
        # For portability we require you to copy the batch id from the first status call:
        # (You can adapt this to parse Location header if your HTTP adapter exposes it.)
        status = client.batch_status_get(resp.get("data", {}).get("id", "")) if "data" in resp else None
        # If you already have a batch id string:
        # batch_id = "2df43e85-ea0f-45c6-bf4a-2baf4d1eaa3c"
        # result = client.poll_batch_until_done(batch_id)
        # df_result = pd.DataFrame(result["data"]) if pd and "data" in result else result
    except Exception as e:
        print("Batch example note (expected if no access):", e)
