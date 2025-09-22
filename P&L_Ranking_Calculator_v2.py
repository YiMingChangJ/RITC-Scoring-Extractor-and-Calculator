"""
Rotman BMO Finance Research and Trading Lab, Uniersity of Toronto (C)
All rights reserved.
"""
"""
CaseRankAnalyzer: Competition Results Aggregator & Ranking Tool

This module processes competition results stored in Excel files. It:
1. Reads team results (P&L, tick-level data, orders/transaction log) from multiple heats and cases.
2. Cleans and normalizes team IDs (handles ETF vs. Volatility naming).
3. Computes and Produces a wide-format results table (per team) with:
   - Team-level P&L aggregation
   - Heat-by-heat rankings
   - Average ranks across heats/cases
   - Tick-level volatility (std) and Sharpe ratios
   - Transaction counts
   - Saves the final results to Excel.

Intended use: automate ranking and award decisions in RITCxCMU competitions.
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from typing import Dict, Tuple, Optional, List # type hints

class CaseRankAnalyzer:
    def __init__(self, main_path: str) -> None:
        self.main_path = main_path
        self.all_data = None
        self.orders_data = None
        self.team_wide = None

    def natural_sort_key(self, path: str) -> Tuple[str, int]:
    # Extract folder name, e.g. "ETF_10"
        folder = os.path.basename(os.path.dirname(path))
        # Extract number (if any)
        match = re.search(r'(\d+)', folder)
        num = int(match.group(1)) if match else float('inf')
        return (re.sub(r'\d+', '', folder), num)  # (case_name, number)

    # Function for natural sort of strings like "ETF_1", "ETF_2", ..., "ETF_10" # Split by non-digit/digit sequences
    def sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def load_and_prepare(self) -> None:
        """Load results, orders."""
        excel_files = sorted(glob.glob(os.path.join(self.main_path, "*", "Results.xlsx")),key=self.natural_sort_key)
        # excel_files = sorted(glob.glob(os.path.join(self.main_path, "*", "Results.xlsx")))
        heat_dfs = [] # List of DataFrames per heat for all of the cases.
        orders_dfs = [] # List of orders DataFrames
        heat_counters = {"ETF": 0, "Volatility": 0, "Unknown": 0}
    
        for file in excel_files:
            folder_name = os.path.basename(os.path.dirname(file))
            if "ETF" in folder_name:
                case = "ETF"
            elif "Volatility" in folder_name:
                case = "Volatility"
            else:
                case = "Unknown"
    
            # === 1. Load results (first sheet) ===
            df = pd.read_excel(file, sheet_name=0)
            df['NLV'] = pd.to_numeric(df['NLV'])
            df['TeamID'] = df['TraderID'].astype(str).str.split('-').str[0]
    
            heat_counters[case] += 1
            heat_name = f"Heat {heat_counters[case]}" # update the Heat number
            df['case'] = case
            df['heat'] = heat_name
            heat_dfs.append(df)
    
            # === 2. Load orders (fourth tab) ===
            orders = pd.read_excel(file, sheet_name="Orders")
            orders['TeamID'] = orders['TraderID'].astype(str).str.split('-').str[0]
            orders_dfs.append(orders)

        # Combine results
        self.all_data = pd.concat(heat_dfs, ignore_index=True)
    
        # Combine all of orders across all heats
        self.orders_data = pd.concat(orders_dfs, ignore_index=True)
    
    # def zero_out_group(self, group: pd.DataFrame) -> pd.DataFrame:
    #     """If any std or sharpe in the group is zero, set all to inf/-inf to exclude the team."""
    #     if (group["std_tick"].eq(0).any()) or (group["sharpe_tick"].eq(0).any()):
    #         group["std_tick"] = np.inf
    #         group["sharpe_tick"] = -np.inf
    #     return group

    def compute_tick_stats(self) -> pd.DataFrame:
        """ Compute tick-by-tick statistics (std and Sharpe ratio) for each team, using the 'Charts - Traders' sheet in Results.xlsx."""
        
        excel_files = sorted(glob.glob(os.path.join(self.main_path, "*", "Results.xlsx")),key=self.natural_sort_key)
        # excel_files = sorted(glob.glob(os.path.join(self.main_path, "*", "Results.xlsx")))
        heat_counters = {"ETF": 0, "Volatility": 0, "Unknown": 0}
        tick_data = []

        for file in excel_files:
            folder_name = os.path.basename(os.path.dirname(file))
            charts = pd.read_excel(file, sheet_name="Charts - Traders") # choose the 'Charts - Traders' sheet only

            if "ETF" in folder_name:
                case = "ETF"
            elif "Volatility" in folder_name:
                case = "Volatility"
            else:
                case = "Unknown"

            heat_counters[case] += 1
            heat_name = f"Heat {heat_counters[case]}" # update the Heat number

            # Select only [NLV] columns
            nlv_cols = [c for c in charts.columns if c.endswith("[NLV]")]

            # Aggregate tick-by-tick by team
            team_map = {c: c.split("-")[0].split(" ")[0] for c in nlv_cols}
            team_series = charts[nlv_cols].groupby(team_map, axis=1).sum()

            for team_id in team_series.columns:
                series = team_series[team_id] # tick-by-tick NLV for this team in this heat
                std_val = series.std() # std return
                mean_val = series.mean() # mean return
                sharpe = mean_val / std_val if std_val != 0 else 0 # sharpe ratio

                tick_data.append({
                    "TeamID": team_id,
                    "case": case,
                    "heat": heat_name,
                    "std_tick": std_val,
                    "sharpe_tick": sharpe
                })

        tick_df = pd.DataFrame(tick_data) # std and sharpe per team/case/heat
        # tick_df = tick_df.groupby(["TeamID", "case"], group_keys=False).apply(self.zero_out_group)
        # Find teams that have a zero-value in ANY heat
        zero_teams = tick_df[(tick_df["std_tick"] == 0.0) | (tick_df["sharpe_tick"] == 0.0)]["TeamID"].unique()

        # Step 2: remove all rows for these teams
        tick_df = tick_df[~tick_df["TeamID"].isin(zero_teams)]

        # rank std (lower is better) and sharpe (higher is better) per case/heat
        tick_df["std_for_rank"] = tick_df["std_tick"].replace({0: np.inf}) 
        # tick_df["std_rank"] = tick_df.groupby(["case", "heat"])["std_for_rank"].transform(lambda s: s.rank(ascending=True, method="dense"))
        tick_df["std_rank"] = tick_df.groupby(["case", "heat"])["std_for_rank"].rank('dense', ascending=True)
        
        tick_df["sharpe_for_rank"] = tick_df["sharpe_tick"].replace({0: -np.inf})
        # tick_df["sharpe_rank"] = tick_df.groupby(["case", "heat"])["sharpe_for_rank"].transform(lambda s: s.rank(ascending=False, method="dense")) # higher Sharpe = better rank 
        tick_df["sharpe_rank"] = tick_df.groupby(["case", "heat"])["sharpe_for_rank"].rank('dense', ascending=False) # higher Sharpe = better rank 
        
        # === Avg rank per case ===
        avg_std_ranks = (
            tick_df.groupby(["TeamID", "case"])
            .agg(avg_rank=("std_rank", "mean"))
            .reset_index()
            )
        avg_sharpe_ranks = (
            tick_df.groupby(["TeamID", "case"])
            .agg(avg_rank=("sharpe_rank", "mean"))
            .reset_index()
            )
        
        # case rank (lower is better)
        avg_std_ranks['case_rank'] = avg_std_ranks.groupby('case')['avg_rank'].rank(method='dense', ascending=True)  

        avg_sharpe_ranks['case_rank'] = avg_sharpe_ranks.groupby('case')['avg_rank'].rank(method='dense', ascending=True)

        # === Case rank + overall rank ===
        avg_std_case_rank = avg_std_ranks.groupby('TeamID')['case_rank'].mean().reset_index()
        avg_std_case_rank = avg_std_case_rank.rename(columns={'case_rank': 'avg_case_rank'})
        avg_std_case_rank['overall_std_rank'] = avg_std_case_rank['avg_case_rank'].rank(method='dense', ascending=True) 

        avg_sharpe_case_rank = avg_sharpe_ranks.groupby('TeamID')['case_rank'].mean().reset_index()
        avg_sharpe_case_rank = avg_sharpe_case_rank.rename(columns={'case_rank': 'avg_case_rank'})
        avg_sharpe_case_rank['overall_sharpe_rank'] = avg_sharpe_case_rank['avg_case_rank'].rank(method='dense', ascending=True)  
        
        # combine std and sharpe ranks
        final_ranks = avg_std_case_rank.merge(avg_sharpe_case_rank, on='TeamID')[['TeamID', 'overall_std_rank', 'overall_sharpe_rank']]

        return final_ranks

    def build_table(self) -> None:
        """Build team-level wide table with P&L, ranks, volatility, and Sharpe from tick-by-tick NLV."""

        # === Replace 0 with -inf for ranking but keep original for stats ===
        self.all_data["NLV_for_rank"] = self.all_data["NLV"].replace({0: -np.inf})

        # === Aggregate P&L to team level (per heat) ===
        team_heat = (
            self.all_data
            .groupby(["TeamID", "case", "heat"])
            .agg(NLV=("NLV", "sum"),NLV_for_rank=("NLV_for_rank", "sum"))
            .reset_index()
        )

        team_heat["NLV_for_rank"] = team_heat["NLV"].replace({0: -np.inf})

        # === Rank teams per case/heat ===
        team_heat["heat_rank"] = team_heat.groupby(["case", "heat"])["NLV_for_rank"].transform(lambda s: s.rank(ascending=False, method="dense")) # the higher the P&L, the better the rank

        # === Pivot P&L and ranks wide ===
        nlv_wide = team_heat.pivot_table(index="TeamID", columns=["case", "heat"], values="NLV", aggfunc="first")
        rank_wide = team_heat.pivot_table(index="TeamID", columns=["case", "heat"], values="heat_rank", aggfunc="first")

        nlv_wide.columns = [f"NLV_{c1}_{c2}" for c1, c2 in nlv_wide.columns]
        rank_wide.columns = [f"Rank_{c1}_{c2}" for c1, c2 in rank_wide.columns]

        # Sort NLV columns
        nlv_wide = nlv_wide.reindex(sorted(nlv_wide.columns, key=self.sort_key), axis=1)
        rank_wide = rank_wide.reindex(sorted(rank_wide.columns, key=self.sort_key), axis=1)

        wide = pd.concat([nlv_wide, rank_wide], axis=1).reset_index()

        # === Avg rank per case ===
        avg_ranks = (
            team_heat.groupby(["TeamID", "case"])
            .agg(avg_rank=("heat_rank", "mean"))
            .reset_index()
            )
        avg_ranks_wide = avg_ranks.pivot_table(
            index="TeamID", columns="case", values="avg_rank"
            ).reset_index()
        avg_ranks_wide = avg_ranks_wide.rename(
            columns={c: f"avg_rank_{c}" for c in avg_ranks_wide.columns if c != "TeamID"}
            )
        wide = wide.merge(avg_ranks_wide, on="TeamID", how="left")

        # === Case rank + overall rank ===
        for col in [c for c in wide.columns if c.startswith("avg_rank_")]:
            case = col.replace("avg_rank_", "")
            wide[f"case_rank_{case}"] = wide[col].rank(method="dense") # before were using min, now using dense to avoid gaps

        case_cols = [c for c in wide.columns if c.startswith("case_rank_")]
        wide["average_case_rank"] = wide[case_cols].mean(axis=1, skipna=True)
        wide["overall_rank"] = wide["average_case_rank"].rank(method="dense") # before were using min, now using dense to avoid gaps

        # === Transaction counts (team-level) ===
        if self.orders_data is not None and not self.orders_data.empty:
            trans_counts = (
                self.orders_data[self.orders_data["Status"] == "TRANSACTED"]
                .groupby("TeamID")
                .size()
                .reset_index(name="transaction_count_rank")
            )
            trans_counts['transaction_count_rank'] = trans_counts['transaction_count_rank'].rank(method='dense', ascending=False) 
            wide = wide.merge(trans_counts, on="TeamID", how="left")

        std_sharpe_rank = self.compute_tick_stats()

        wide = wide.merge(std_sharpe_rank, on='TeamID', how="left")
        # wide["SharpeRank"] = wide["SharpeRank"].fillna(0)  
        self.team_wide = wide

    def save(self, filename: str) -> None:
        """Save team-level table to Excel."""
        out_file = os.path.join(self.main_path, filename)
        self.team_wide.to_excel(out_file, index=False)
        print(f"Saved {out_file}")

# === Example Usage ===
if __name__ == "__main__":
    main_path = r"C:\Users\yiming.chang\OneDrive - University of Toronto\Desktop\Yi-Ming Chang\Educational Developer\RITCx\RITCxCMU 2025\RITCxCMU2025-Competition Results"
    analyzer = CaseRankAnalyzer(main_path)
    analyzer.load_and_prepare()
    analyzer.build_table()
    analyzer.save("RITCxCMU2025-Team_Results.xlsx")
