#!/usr/bin/env python3
"""
Rotman BMO Finance Research and Trading Lab, University of Toronto (C)
All rights reserved.

CaseRankAnalyzer — Heat-Accurate Team Ranking (Complete Script)

Purpose
-------
Fixes the "values under wrong heat column" problem by:
1) Extracting the numeric heat from each subfolder name (e.g., ".../LT3_Heat 10/Results.xlsx" -> heat_num=10)
2) Using that numeric heat to label rows and to order pivot columns
3) (Optional) De-duplicating multiple files for the same (case, heat_num) by keeping the newest
4) Ranking AFTER team aggregation, and setting NLV_for_rank = -inf only when a team's aggregated NLV == 0

Outputs
-------
- Wide table with columns like:
  TeamID, NLV_LT3_Heat 1, ..., NLV_LT3_Heat 11, Rank_LT3_Heat 1, ..., Rank_LT3_Heat 11,
  avg_rank_LT3, case_rank_LT3, average_case_rank, overall_rank

Notes
-----
- Column names for heats remain correct and stable — only data placement is corrected.
- Save is robust against Windows file locks (Excel/OneDrive) and too-long paths.
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# ----------------------- Path utilities -----------------------
def _shorten_path(p: str, max_len: int = 230) -> str:
    """If Windows path is too long, write to %TEMP% with same filename."""
    p_obj = Path(p)
    if len(str(p_obj)) <= max_len:
        return str(p_obj)
    return str(Path(os.getenv("TEMP", ".")) / p_obj.name)


def _timestamped(name: str, suffix: str = ".xlsx") -> str:
    """Insert a timestamp before the suffix (e.g., results_20251108-091501.xlsx)."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = Path(name).stem
    return f"{stem}_{ts}{suffix}"


# ======================= Analyzer =======================
class CaseRankAnalyzer:
    def __init__(self, main_path: str) -> None:
        self.main_path = main_path
        self.all_data: Optional[pd.DataFrame] = None
        self.team_wide: Optional[pd.DataFrame] = None

    # ---------- helpers ----------
    @staticmethod
    def _extract_case(folder_name: str) -> str:
        low = folder_name.lower()
        if "lt3" in low:
            return "LT3"
        if "volatility" in low:
            return "Volatility"
        return "Unknown"

    @staticmethod
    def _extract_heat_num(folder_name: str) -> Optional[int]:
        """
        Prefer explicit 'heat N'; otherwise use the LAST integer found in the folder name.
        Examples that work:
          'LT3_Heat 8' -> 8
          'LT3-H9' -> 9
          'LT3_11' -> 11
          'Heat_02' -> 2
        """
        m = re.search(r"heat[\s_\-]*?(\d+)", folder_name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        ints = re.findall(r"(\d+)", folder_name)
        if ints:
            return int(ints[-1])
        return None

    @staticmethod
    def _pick_team_series(df: pd.DataFrame) -> pd.Series:
        """Derive TeamID from TraderID or TeamID (prefix before '-')."""
        low = {str(c).strip().lower(): c for c in df.columns}
        if "traderid" in low:
            base = df[low["traderid"]].astype(str)
        elif "teamid" in low:
            base = df[low["teamid"]].astype(str)
        else:
            raise KeyError("Neither 'TraderID' nor 'TeamID' column found in results sheet.")
        return base.str.split("-").str[0]

    @staticmethod
    def _coerce_money_like(s: pd.Series) -> pd.Series:
        """Coerce money-like values ($, commas, spaces, and (negatives)) to float."""
        s = s.astype(str).str.replace(r"\s+", "", regex=True)
        s = s.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    @staticmethod
    def _pick_nlv_series(df: pd.DataFrame) -> pd.Series:
        """Select the NLV column robustly; fall back to PnL if necessary."""
        low = {str(c).strip().lower(): c for c in df.columns}
        if "nlv" in low:
            s = df[low["nlv"]]
        else:
            cands = [k for k in low if re.search(r"\bnlv\b", k)]
            if cands:
                s = df[low[sorted(cands, key=len)[0]]]
            else:
                cands = [k for k in low if re.search(r"\bp&?nl\b", k)]
                if not cands:
                    raise KeyError("Could not locate NLV / PnL column in results sheet.")
                s = df[low[sorted(cands, key=len)[0]]]
        return CaseRankAnalyzer._coerce_money_like(s)

    # ---------- load with strict heat mapping ----------
    def load_and_prepare(self) -> None:
        """
        1) Collect files:
             - Prefer .../<HeatFolder>/Results.xlsx
             - If none, fall back to .../<HeatFolder>/*.xlsx (first sheet)
        2) Extract case and numeric heat from each folder
        3) Build rows with exact heat_num and label 'Heat {heat_num}'
        4) (Optional) Deduplicate (case, heat_num) by keeping the newest file
        """
        files = glob.glob(os.path.join(self.main_path, "*", "Results.xlsx"))
        if not files:
            files = glob.glob(os.path.join(self.main_path, "*", "*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No .xlsx files found under {self.main_path!r}.")

        # Choose newest per (case, heat_num) to avoid collisions
        chosen: Dict[Tuple[str, int], str] = {}
        for fp in files:
            folder = os.path.basename(os.path.dirname(fp))
            case = self._extract_case(folder)
            heat_num = self._extract_heat_num(folder)
            if heat_num is None:
                # Skip folders we cannot map to a heat number to avoid misplacement
                continue
            key = (case, heat_num)
            if key not in chosen or os.path.getmtime(fp) > os.path.getmtime(chosen[key]):
                chosen[key] = fp

        if not chosen:
            raise ValueError("No usable heat numbers could be extracted from folder names.")

        rows: List[pd.DataFrame] = []
        for (case, heat_num), fp in sorted(chosen.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            df = pd.read_excel(fp, sheet_name=0)
            team_series = self._pick_team_series(df)
            nlv_series = self._pick_nlv_series(df)

            rows.append(pd.DataFrame({
                "TeamID": team_series,
                "case": case,
                "heat_num": int(heat_num),               # numeric key for ordering
                "heat": f"Heat {int(heat_num)}",         # pretty label
                "NLV": nlv_series,
            }))

        self.all_data = pd.concat(rows, ignore_index=True)
        if self.all_data.empty:
            raise ValueError("Loaded data is empty after concatenation.")

    # ---------- build table with correct heat alignment ----------
    def build_table(self) -> None:
        if self.all_data is None or self.all_data.empty:
            raise ValueError("No data loaded. Call load_and_prepare() first.")

        df = self.all_data.copy()
        df["NLV"] = pd.to_numeric(df["NLV"], errors="coerce").fillna(0.0)

        # Aggregate to team per (case, heat_num)
        team_heat = (
            df.groupby(["TeamID", "case", "heat_num"], as_index=False)
              .agg(NLV=("NLV", "sum"))
        )
        team_heat["heat"] = "Heat " + team_heat["heat_num"].astype(int).astype(str)

        # Rank rule: AFTER aggregation → -inf if team NLV == 0 for that heat
        team_heat["NLV_for_rank"] = np.where(team_heat["NLV"] == 0.0, -np.inf, team_heat["NLV"])

        # Dense rank within each (case, heat_num): higher NLV is better (1 is best)
        team_heat["heat_rank"] = (
            team_heat.groupby(["case", "heat_num"], group_keys=False)["NLV_for_rank"]
                     .rank(method="dense", ascending=False)
        )

        # ----- Pivot with strict numeric ordering -----
        # Use MultiIndex (case, heat_num) → guaranteed order; then flatten names
        nlv_wide = team_heat.pivot_table(
            index="TeamID", columns=["case", "heat_num"], values="NLV", aggfunc="first"
        ).sort_index(axis=1, level=[0, 1])

        rank_wide = team_heat.pivot_table(
            index="TeamID", columns=["case", "heat_num"], values="heat_rank", aggfunc="first"
        ).sort_index(axis=1, level=[0, 1])

        # Flatten with your expected naming scheme: NLV_<case>_Heat <num>
        nlv_wide.columns = [f"NLV_{case}_Heat {int(h)}" for (case, h) in nlv_wide.columns.to_list()]
        rank_wide.columns = [f"Rank_{case}_Heat {int(h)}" for (case, h) in rank_wide.columns.to_list()]

        wide = pd.concat([nlv_wide, rank_wide], axis=1).reset_index()

        # Average heat ranks per case (using numeric heat underneath)
        avg_ranks = (
            team_heat.groupby(["TeamID", "case"], as_index=False)["heat_rank"].mean()
                     .rename(columns={"heat_rank": "avg_rank"})
        )
        avg_ranks_wide = avg_ranks.pivot_table(index="TeamID", columns="case", values="avg_rank").reset_index()
        avg_ranks_wide = avg_ranks_wide.rename(
            columns={c: f"avg_rank_{c}" for c in avg_ranks_wide.columns if c != "TeamID"}
        )

        wide = wide.merge(avg_ranks_wide, on="TeamID", how="left")

        # Case ranks and overall rank (lower is better)
        for col in [c for c in wide.columns if c.startswith("avg_rank_")]:
            wide[f"case_rank_{col.replace('avg_rank_', '')}"] = wide[col].rank(method="dense", ascending=True)

        case_cols = [c for c in wide.columns if c.startswith("case_rank_")]
        if case_cols:
            wide["average_case_rank"] = wide[case_cols].mean(axis=1, skipna=True)
            wide["overall_rank"] = wide["average_case_rank"].rank(method="dense", ascending=True)
        else:
            wide["average_case_rank"] = np.nan
            wide["overall_rank"] = np.nan

        self.team_wide = wide

    # ---------- robust save ----------
    def save(self, filename: str) -> None:
        """
        Save results to Excel.
        - Handles Windows file locks (saves timestamped copy if locked)
        - Guards against too-long paths (writes to TEMP)
        - Falls back to TEMP on path errors
        """
        if self.team_wide is None or self.team_wide.empty:
            raise ValueError("No computed table to save. Call build_table() first.")

        out_file = filename if os.path.isabs(filename) else os.path.join(self.main_path, filename)
        out_file = _shorten_path(out_file)

        try:
            d = os.path.dirname(out_file)
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass

        # Try preferred engine; fall back gracefully if not installed
        try:
            self.team_wide.to_excel(out_file, index=False, engine="xlsxwriter")
            print(f"Saved {out_file}")
            return
        except ImportError:
            try:
                self.team_wide.to_excel(out_file, index=False)  # let pandas choose engine
                print(f"Saved {out_file}")
                return
            except PermissionError:
                pass
        except PermissionError:
            pass

        # If we got here, it's likely a lock → timestamped sibling
        try:
            alt_file = os.path.join(os.path.dirname(out_file) or ".", _timestamped(os.path.basename(out_file)))
            try:
                self.team_wide.to_excel(alt_file, index=False, engine="xlsxwriter")
                print(f"Target file locked. Saved a new copy instead: {alt_file}")
                return
            except ImportError:
                self.team_wide.to_excel(alt_file, index=False)
                print(f"Target file locked. Saved a new copy instead: {alt_file}")
                return
        except Exception:
            # Final fallback: TEMP
            temp_file = os.path.join(os.getenv("TEMP", "."), _timestamped(os.path.basename(out_file)))
            try:
                self.team_wide.to_excel(temp_file, index=False, engine="xlsxwriter")
            except ImportError:
                self.team_wide.to_excel(temp_file, index=False)
            print(f"Saved to TEMP due to lock/path issue: {temp_file}")


# ========================== Runner ==========================
if __name__ == "__main__":
    # Set your main directory containing subfolders with Results.xlsx
    main_path = r"C:\Users\yiming.chang\OneDrive - University of Toronto\Desktop\Yi-Ming Chang\Educational Developer\RITC\RITCxTCP 2025\Commpetition results"

    analyzer = CaseRankAnalyzer(main_path)
    analyzer.load_and_prepare()
    analyzer.build_table()
    analyzer.save("RITCxTCP2025-Team_Results.xlsx")
