#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RITC Scoring — Teams-Only with Subheat Pivots (folder-based, SQL-accurate ranks)

Inputs (each CSV in case subfolders):
  TraderID, FirstName, LastName, NLV

Outputs (Excel workbook sheets):
  - SubHeatRanksStudent   (TeamCode-level: Profit & Rank per subheat)
  - HeatRanksStudent      (TeamCode-level)
  - CaseRanksStudent      (TeamCode-level; Score = TeamCount - Rank + 1)
  - TotalRanksStudent     (TeamCode-level; weighted sum across cases, VAR, Rank)
  - PnL_<case>            (wide pivot: teams × H#S# → Profit)
  - Rank_<case>           (wide pivot: teams × H#S# → Rank)

Folder layout example:
  ROOT/
    1 BP Commodities/
      H1SH1.csv, H1SH2.csv, ...
    2 Flow Traders ETF/
      ...
    3 Bridgewater Fixed Income/
      ...
    4 Matlab Volatility/
      ...

Usage:
  python scoring_views_teams_only_pivots.py --root "<path>" --out "ScoresCheck_Teams.xlsx"
  python scoring_views_teams_only_pivots.py --root "<path>" --out "ScoresCheck_Teams.xlsx" \
     --weights "Commodities::30,ETF::25,Fixed Income::20,Volatility::25"
"""

from __future__ import annotations
import argparse
import csv
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# -------- Config --------

DEFAULT_CASE_WEIGHTS = {
    "Commodities": 25.0,
    "Flow Traders ETF": 25.0,
    "Fixed Income": 25.0,
    "Volatility": 25.0,
}

HEAT_PAT = re.compile(r"(?:^|[^a-z])(heat|h)[ _-]*([0-9]+)(?=\D|$)", re.IGNORECASE)
SUB_PAT  = re.compile(r"(?:^|[^a-z])(sub|s)[ _-]*([0-9]+)(?=\D|$)",  re.IGNORECASE)

ENCODING_TRY = [
    "utf-8", "utf-8-sig",
    "cp1252", "latin-1", "iso-8859-1",
    "utf-16", "utf-16-le", "utf-16-be",
]
DELIMS_TRY = [",", "\t", ";", None]  # None => engine='python' autodetect

# -------- Helpers --------

def parse_weights_arg(arg: Optional[str]) -> Dict[str, float]:
    if not arg:
        return DEFAULT_CASE_WEIGHTS.copy()
    out: Dict[str, float] = {}
    for pair in arg.split(","):
        if "::" in pair:
            k, v = pair.split("::", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out or DEFAULT_CASE_WEIGHTS.copy()

def case_weight_for(name: str, weights_map: Dict[str, float]) -> float:
    for key, w in weights_map.items():
        if key.lower() in name.lower():
            return float(w)
    return float(DEFAULT_CASE_WEIGHTS.get(name, 25.0))

def read_csv_any(path: Path, verbose: bool = True) -> pd.DataFrame:
    # Fast path
    try:
        df = pd.read_csv(path)
        if verbose:
            print(f"    ✓ {path.name} decoded as utf-8 sep=','")
        return df
    except Exception:
        pass

    data = path.read_bytes()
    last_err = None
    for enc in ENCODING_TRY:
        for sep in DELIMS_TRY:
            try:
                if enc.startswith("utf-16"):
                    if sep is None:
                        for sep2 in [",", "\t", ";"]:
                            try:
                                df = pd.read_csv(BytesIO(data), encoding=enc, sep=sep2)
                                if verbose:
                                    print(f"    ✓ {path.name} decoded as {enc} sep='{sep2}'")
                                return df
                            except Exception:
                                continue
                        continue
                    else:
                        df = pd.read_csv(BytesIO(data), encoding=enc, sep=sep)
                        if verbose:
                            print(f"    ✓ {path.name} decoded as {enc} sep='{sep}'")
                        return df
                else:
                    if sep is None:
                        df = pd.read_csv(BytesIO(data), encoding=enc, sep=None, engine="python")
                        if verbose:
                            print(f"    ✓ {path.name} decoded as {enc} sep='auto'")
                        return df
                    else:
                        df = pd.read_csv(BytesIO(data), encoding=enc, sep=sep)
                        if verbose:
                            print(f"    ✓ {path.name} decoded as {enc} sep='{sep}'")
                        return df
            except Exception as e:
                last_err = e
                continue
    # Fallback: decode with replacements + sniff delimiter
    text = data.decode("utf-8", errors="replace")
    try:
        sniff = csv.Sniffer().sniff(text.splitlines()[0])
        sep = sniff.delimiter
    except Exception:
        sep = ","
    df = pd.read_csv(StringIO(text), sep=sep)
    if verbose:
        print(f"    ✓ {path.name} decoded via 'replace' sep='{sep}' (fallback)")
    return df

def infer_heat_sub(text: str) -> Tuple[Optional[int], Optional[int]]:
    h = s = None
    m = HEAT_PAT.search(text)
    if m: h = int(m.group(2))
    m = SUB_PAT.search(text)
    if m: s = int(m.group(2))
    return h, s

def safe_sheet_name(name: str, prefix: str = "", used: Optional[set] = None) -> str:
    """
    Excel sheet name <= 31 chars, strip invalid chars, ensure uniqueness.
    """
    invalid = set(r'[]:*?/\\')
    base = "".join(ch for ch in name if ch not in invalid).strip()
    if prefix:
        base = f"{prefix}_{base}"
    if len(base) > 31:
        base = base[:31]
    if used is None:
        return base
    s = base
    i = 1
    while s in used:
        suffix = f"_{i}"
        s = (base[:31-len(suffix)] + suffix)
        i += 1
    used.add(s)
    return s

# -------- Scan folders → base rows (TeamCode only) --------

def scan_root(root: Path, weights_map: Dict[str, float]):
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder not found: {root}")

    case_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not case_dirs:
        raise RuntimeError(f"No case folders under: {root}")

    print("📂 Cases found:")
    for d in sorted(case_dirs, key=lambda p: p.name.lower()):
        print(f"  - {d.name}")

    rows: List[dict] = []
    for case_dir in sorted(case_dirs, key=lambda p: p.name.lower()):
        case_name = case_dir.name.strip()
        weight = case_weight_for(case_name, weights_map)

        files = sorted([p for p in case_dir.rglob("*.csv")], key=lambda p: str(p).lower())
        if not files:
            print(f"⚠️  {case_name}: no CSV files — skipping")
            continue

        print(f"→ {case_name}: {len(files)} file(s)")

        next_heat = 1
        for f in files:
            rel = str(f.relative_to(case_dir)).replace("\\", "/")
            ui_heat, ui_sub = infer_heat_sub(rel)
            if ui_heat is None:
                ui_heat = next_heat
                next_heat += 1
            if ui_sub is None:
                ui_sub = 1

            try:
                df = read_csv_any(f, verbose=True)
            except Exception as ex:
                print(f"    ⚠️ {f.name}: read error -> {ex} (skipped)")
                continue

            expected = ["TraderID", "FirstName", "LastName", "NLV"]
            missing = [c for c in expected if c not in df.columns]
            if missing:
                print(f"    ⚠️ {f.name}: missing columns {missing} (skipped)")
                continue

            # Normalize core fields
            df["TraderID"] = df["TraderID"].astype(str)
            df["NLV"] = pd.to_numeric(df["NLV"], errors="coerce").fillna(0.0)

            # TeamCode = prefix of TraderID before first '-'
            team_code = df["TraderID"].str.split("-", n=1).str[0].fillna("TRADERS")
            team_code = team_code.where(team_code.str.len() > 0, "TRADERS")

            # Emit rows (we only need TeamCode + NLV; Adjustment=0)
            for i in range(len(df)):
                rows.append({
                    "CaseName":  case_name,
                    "CaseID":    abs(hash(case_name)) % 1_000_000,
                    "HeatID":    int(ui_heat),
                    "SubHeatID": int(ui_sub),
                    "TeamCode":  str(team_code.iat[i]),
                    "NLV":       float(df["NLV"].iat[i]),
                    "Adjustment": 0.0,
                    "Weight":     float(weight),
                    "Publish":    1,     # enforced
                    "Type":       1,     # enforced
                })

    if not rows:
        raise RuntimeError("No rows loaded. Check folder contents and file columns.")

    base = pd.DataFrame(rows)
    return base

# -------- Views (TeamCode-only, exact SQL semantics) --------

def view_AllPnLStudent(base: pd.DataFrame) -> pd.DataFrame:
    allp = base[(base["Publish"] == 1) & (base["Type"] == 1)].copy()
    allp["Profit"] = allp["NLV"] + allp["Adjustment"]
    cols = ["CaseID","CaseName","HeatID","SubHeatID","TeamCode","Profit","Weight"]
    return allp[cols].copy()

def view_SubHeatRanksStudent(allp: pd.DataFrame) -> pd.DataFrame:
    sub = (
        allp.groupby(["CaseID","CaseName","HeatID","SubHeatID","TeamCode"], as_index=False)
            .agg(Profit=("Profit","sum"),
                 Weight=("Weight","min"))
    )
    # zeros last, then Profit DESC
    sub["_zero_key"] = np.where(sub["Profit"] != 0.0, 0, 1)

    ranks = np.empty(len(sub), dtype=int)
    for _, idx in sub.groupby(["CaseID","HeatID","SubHeatID"]).groups.items():
        part = sub.loc[idx].copy()
        part.sort_values(["_zero_key","Profit"], ascending=[True, False], inplace=True)
        current = 1
        rser = pd.Series(index=part.index, dtype=int)
        for _, same in part.groupby("Profit", sort=False):
            rser.loc[same.index] = current
            current += len(same)
        ranks[idx] = rser.loc[idx].values

    sub["Rank"] = ranks.astype(int)
    sub["Score"] = sub["Rank"].astype(float)

    out = sub.drop(columns=["_zero_key"])
    return out[[
        "Profit","TeamCode","CaseID","CaseName","HeatID","SubHeatID","Weight","Rank","Score"
    ]].sort_values(["CaseID","HeatID","SubHeatID","Rank","TeamCode"]).reset_index(drop=True)

def view_HeatRanksStudent(subheat: pd.DataFrame) -> pd.DataFrame:
    agg = (
        subheat.groupby(["CaseID","CaseName","HeatID","TeamCode"], as_index=False)
               .agg(Score=("Score","mean"),
                    Weight=("Weight","min"))
    )
    heat_ranks = np.empty(len(agg), dtype=int)
    for _, idx in agg.groupby(["CaseID","HeatID"]).groups.items():
        part = agg.loc[idx].copy()
        part.sort_values(["Score"], ascending=[True], inplace=True)
        current = 1
        rser = pd.Series(index=part.index, dtype=int)
        for _, same in part.groupby("Score", sort=False):
            rser.loc[same.index] = current
            current += len(same)
        heat_ranks[idx] = rser.loc[idx].values
    agg["Rank"] = heat_ranks.astype(int)

    return agg[[
        "TeamCode","CaseID","CaseName","HeatID","Weight","Score","Rank"
    ]].sort_values(["CaseID","HeatID","Rank","TeamCode"]).reset_index(drop=True)

def view_CaseRanksStudent(heat: pd.DataFrame, team_codes: pd.Series) -> pd.DataFrame:
    team_count = int(team_codes.nunique())

    avg_heat = (
        heat.groupby(["TeamCode","CaseID","CaseName"], as_index=False)
            .agg(AvgHeatRank=("Rank","mean"),
                 Weight=("Weight","min"))
    )

    cr_ranks = np.empty(len(avg_heat), dtype=int)
    for _, idx in avg_heat.groupby("CaseID").groups.items():
        part = avg_heat.loc[idx].copy()
        part.sort_values(["AvgHeatRank"], ascending=[True], inplace=True)
        current = 1
        rser = pd.Series(index=part.index, dtype=int)
        for _, same in part.groupby("AvgHeatRank", sort=False):
            rser.loc[same.index] = current
            current += len(same)
        cr_ranks[idx] = rser.loc[idx].values
    avg_heat["Rank"] = cr_ranks.astype(int)

    avg_heat["Score"] = team_count - avg_heat["Rank"] + 1

    return avg_heat[[
        "TeamCode","CaseID","CaseName","Weight","Score","Rank"
    ]].sort_values(["CaseID","Rank","TeamCode"]).reset_index(drop=True)

def view_TotalRanksStudent(case_ranks: pd.DataFrame) -> pd.DataFrame:
    cr = case_ranks.copy()
    cr["Weighted"] = cr["Score"] * (cr["Weight"] / 100.0)

    totals = (
        cr.groupby("TeamCode", as_index=False)
          .agg(Score=("Weighted","sum"),
               Var=("Score", lambda s: float(pd.Series(s).var(ddof=1)) if len(s) >= 2 else np.nan))
    )

    var_for_rank = totals["Var"].fillna(np.inf)
    order = np.lexsort((var_for_rank.values, -totals["Score"].values))
    key = list(zip(totals["Score"].round(12), var_for_rank.round(12)))

    rank_map: Dict[Tuple[float,float], int] = {}
    cur = 1
    for idx in order:
        k = key[idx]
        if k not in rank_map:
            rank_map[k] = cur
        cur += 1

    totals["Rank"] = [rank_map[k] for k in key]
    return totals[["TeamCode","Score","Var","Rank"]].sort_values(["Rank","TeamCode"]).reset_index(drop=True)

# -------- Extra: per-case wide pivots (PnL & Rank per subheat) --------

def build_case_pivots(subheat: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Returns: { case_name: { "PnL": pnl_pivot_df, "Rank": rank_pivot_df } }
    Columns labeled 'H{HeatID}S{SubHeatID}', index = TeamCode
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    sub = subheat.copy()
    sub["H_S"] = sub.apply(lambda r: f"H{int(r.HeatID)}S{int(r.SubHeatID)}", axis=1)

    for case_name, g in sub.groupby("CaseName"):
        pnl_piv = g.pivot_table(index="TeamCode", columns="H_S", values="Profit", aggfunc="first")
        rnk_piv = g.pivot_table(index="TeamCode", columns="H_S", values="Rank",   aggfunc="first")
        # Sort columns by numeric heat, sub
        cols_sorted = sorted(pnl_piv.columns, key=lambda k: (int(k.split('S')[0][1:]), int(k.split('S')[1])))
        pnl_piv = pnl_piv.reindex(columns=cols_sorted).sort_index()
        rnk_piv = rnk_piv.reindex(columns=cols_sorted).sort_index()
        out[case_name] = {"PnL": pnl_piv, "Rank": rnk_piv}
    return out

# -------- Pipeline --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing per-case subfolders")
    ap.add_argument("--out",  default="ScoresCheck_Teams.xlsx", help="Output Excel workbook")
    ap.add_argument("--weights", default=None,
                    help='Optional mapping: "Commodities::25,ETF::25,Fixed Income::25,Volatility::25" (substring match)')
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    weights_map = parse_weights_arg(args.weights)

    # 1) Scan to base (TeamCode-only)
    base = scan_root(root, weights_map)
    print(f"\n✅ Loaded trader rows: {len(base)} | Cases: {base['CaseID'].nunique()} | Teams: {base['TeamCode'].nunique()}")

    # 2) Views (exact SQL math, adapted to TeamCode)
    allp = view_AllPnLStudent(base)
    sub  = view_SubHeatRanksStudent(allp)  # contains Profit & Rank per subheat/team
    heat = view_HeatRanksStudent(sub)
    case = view_CaseRanksStudent(heat, team_codes=base["TeamCode"])
    total= view_TotalRanksStudent(case)

    # 3) Build per-case pivots
    pivots = build_case_pivots(sub)

    # 4) Save workbook
    used_names = set()
    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        sub.to_excel(xl,   index=False, sheet_name="SubHeatRanksStudent")
        heat.to_excel(xl,  index=False, sheet_name="HeatRanksStudent")
        case.to_excel(xl,  index=False, sheet_name="CaseRanksStudent")
        total.to_excel(xl, index=False, sheet_name="TotalRanksStudent")

        # Add pivots per case (sanitize sheet names)
        for case_name, d in pivots.items():
            pnl_sheet  = safe_sheet_name(case_name, prefix="PnL",  used=used_names)
            rank_sheet = safe_sheet_name(case_name, prefix="Rank", used=used_names)
            d["PnL"].to_excel(xl,  sheet_name=pnl_sheet)
            d["Rank"].to_excel(xl, sheet_name=rank_sheet)

    print(f"📄 Wrote: {args.out}")
    print("\nExample — first 10 SubHeat rows (Profit & Rank per team/subheat):")
    print(sub.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
