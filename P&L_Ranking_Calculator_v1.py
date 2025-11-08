"""
Rotman BMO Finance Research and Trading Lab, Uniersity of Toronto (C)
All rights reserved.
"""
#%%
import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, Tuple, Optional, List # type hints


class CaseRankAnalyzer:
    def __init__(self, main_path: str) -> None:
        self.main_path = main_path
        self.all_data = None
        self.wide = None

    def load_and_prepare(self) -> None:
        """Load all heat results, clean data, assign case & heat numbers, compute heat ranks."""
        excel_files = sorted(glob.glob(os.path.join(self.main_path, "*", "Results.xlsx"))) # All excel files paths
        heat_dfs = [] # List of DataFrames per heat for all of the cases.
        heat_counters = {"LT3": 0, "Volatility": 0, "Unknown": 0} # Count heats per case

        for file in excel_files:
            df = pd.read_excel(file)

            # Ensure numeric NLV
            df['NLV'] = pd.to_numeric(df['NLV']) #, errors='coerce')
            # Replace 0 with -inf = did not participate
            df['NLV'] = df['NLV'].replace({0: -np.inf}) # -1000000: -np.inf})

            df['root'] = df['TraderID'].astype(str).str.split('-').str[0]

            df['FirstName'] = df['FirstName'].fillna('Unknown')
            df['LastName'] = df['LastName'].fillna('Unknown')

            folder_name = os.path.basename(os.path.dirname(file)) # extract a;; pf 
            if "LT3" in folder_name:
                case = "LT3"
            elif "Volatility" in folder_name:
                case = "Volatility"
            else:
                case = "Unknown"

            heat_counters[case] += 1
            heat_name = f"Heat {heat_counters[case]}"

            df['case'] = case
            df['heat'] = heat_name
            heat_dfs.append(df)

        # Combine all heats
        self.all_data = pd.concat(heat_dfs, ignore_index=True)

        # Rank per case + heat (dense: no gaps, ties → same rank), groupby case and heat, then rank the NLV within each group
        self.all_data['heat_rank'] = self.all_data.groupby(['case','heat'])['NLV'] \
            .transform(lambda s: s.rank(ascending = False, method = 'dense'))

    def build_table(self) -> None:
        """Build per-trader wide table with NLV, ranks, avg ranks, case ranks, and overall rank."""
        all_data = self.all_data

        # Create two pivot tables from all_data: NLV and Rank per heat, grouped by case
        nlv_wide = all_data.pivot_table(
            index=['TraderID','FirstName','LastName','root'], # Each row represents a unique trader (with their ID, name, and team root).
            columns=['case','heat'], values='NLV', aggfunc='first' # for each case and each heat inside that case,
        )
        rank_wide = all_data.pivot_table(
            index=['TraderID','FirstName','LastName','root'],
            columns=['case','heat'], values='heat_rank', aggfunc='first'
        )

        # Flatten column names, loop through two levels of the multiIndex case and heat, builds a flat string like NLV_ETF_Heat1 or Rank_ETF_Heat1
        nlv_wide.columns = [f"NLV_{c1}_{c2}" for c1,c2 in nlv_wide.columns]
        rank_wide.columns = [f"Rank_{c1}_{c2}" for c1,c2 in rank_wide.columns]

        wide = pd.concat([nlv_wide, rank_wide], axis=1).reset_index() # Combine NLV and Rank tables side by side form a table

        # ---- Compute avg_rank per case ----
        avg_ranks = (
            all_data.groupby(['TraderID','FirstName','LastName','root','case'])
            .agg(avg_rank=('heat_rank','mean'))
            .reset_index()
        )
        avg_ranks_wide = avg_ranks.pivot_table(
            index=['TraderID','FirstName','LastName','root'],
            columns='case', values='avg_rank'
        ).reset_index()

        avg_ranks_wide = avg_ranks_wide.rename(
            columns={c: f"avg_rank_{c}" for c in avg_ranks_wide.columns if c not in ['TraderID','FirstName','LastName','root']}
        )

        wide = wide.merge(avg_ranks_wide, on=['TraderID','FirstName','LastName','root'], how='left')

        # ---- Case rank (within each case) ----
        for col in [c for c in wide.columns if c.startswith("avg_rank_")]:
            case = col.replace("avg_rank_","")
            wide[f"case_rank_{case}"] = wide[col].rank(method='min')

        # ---- Overall rank ----
        case_cols = [c for c in wide.columns if c.startswith("case_rank_")]
        wide['average_case_rank'] = wide[case_cols].mean(axis=1, skipna=True)
        wide['overall_rank'] = wide['average_case_rank'].rank(method='min')

        self.wide = wide

    def save(self, filename: str) -> None:
        """Save the wide table to Excel."""
        if self.wide is not None:
            out_file = os.path.join(self.main_path, filename)
            self.wide.to_excel(out_file, index=False)
            print(f" Saved {out_file}")
        else:
            print("No table to save. Run build_wide_table() first.")


# === Example Usage ===
if __name__ == "__main__":
    main_path = r"C:\Users\yiming.chang\OneDrive - University of Toronto\Desktop\Yi-Ming Chang\Educational Developer\RITC\RITCxTCP 2025\Commpetition results"
    analyzer = CaseRankAnalyzer(main_path)
    analyzer.load_and_prepare()
    analyzer.build_table()
    analyzer.save("RITCxTCP2025-Practice_Session_Results.xlsx")

# %%
