This is a massive and highly impressive suite of tools you have built! You have essentially created a complete offline shadow-scoring engine that replicates, validates, and extends the capabilities of the primary RITC web portal.

Because your repository now contains six distinct scripts serving different phases of the competition (practice, validation, live scoring, and adjustments), the `README.md` needs to act as a directory so future lab assistants or developers know exactly which script to run for which scenario.

Here is a completely overhauled, highly professional `README.md` that incorporates all of your new scripts, links, and detailed explanations of the logic.

---

# 🏆 RITC & RITCx Scoring Extractor and Calculator Suite

## 📖 Overview

This repository contains a comprehensive suite of Python scripts designed to analyze, rank, validate, and summarize competition results for the BMO Finance Research and Trading Lab.

These tools handle everything from cross-checking the official RITC Web Scoring Portal to generating standalone leaderboards for RITCx, ROTC, and High School challenges. The suite is designed to handle messy data, automatically penalize non-participating teams, and generate role-specific practice leaderboards.

---

## 🧰 The Tool Suite (Script Directory)

### 1. [Core Scoring Engine (`ScoringPortal.py`)](https://www.google.com/search?q=%5Bhttps://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/ScoringPortal.py%5D(https://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/ScoringPortal.py))

**Purpose:** Web Portal Validation & Cross-Checking.
This script acts as an offline shadow-scoring engine. It parses raw CSV result files and replicates the exact SQL-based ranking semantics used by the live RITC Web Scoring portal. It is used to cross-check and validate that the web portal's output matches the raw data exactly, ensuring 100% accuracy during high-stakes competitions.

### 2. [P&L Ranking Calculators (The CaseRankAnalyzer Suite)](https://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%20and%20RITC%20Scoring/P%26L_Ranking_Calculator_Team.py)

*(Includes both `P&L_Ranking_Calculator_Team.py` and `P&L_Ranking_Calculator_Trader.py`)*
**Purpose:** Full Competition Scoring for RITCx, ROTC, and High School events.
These scripts process raw `Results.xlsx` heat files from cases like ETF Arbitrage or Volatility Trading.

* Extracts Net Liquidation Value (NLV).
* **Missing Round Logic:** Automatically identifies teams/traders that missed early heats, densifies the data grid, fills their NLV with `0`, and assigns them a `-inf` score so they are accurately ranked last for that specific heat.
* Generates wide-table outputs showing NLV, Heat Ranks, Average Case Ranks, and Overall Competition Ranks side-by-side.

### 3. [Role-Based Leaderboards (`GBE_Energy_Practice.py`)](https://www.google.com/search?q=%5Bhttps://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/GBE_Energy_Practice.py%5D(https://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/GBE_Energy_Practice.py))

**Purpose:** Practice Session Teasers.
During practice sessions, it is important to motivate participants without revealing the entire leaderboard. This script reads a single result file, parses the `TraderID` tags (e.g., T1, D, P), and generates isolated **Top 10 Leaderboards** for specific roles (Traders, Distributors, Producers) and Top 10 Teams Overall.

### 4. [Social Outcry Ranking (`SocialOutry_Ranking_v2.py`)](https://www.google.com/search?q=%5Bhttps://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/SocialOutry_Ranking_v2.py%5D(https://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/SocialOutry_Ranking_v2.py))

**Purpose:** Quant & Social Outcry Validation.
Validates the results of the Quant or Social Outcry cases. It ingests raw trade ledger data downloaded via Admin privileges from the web portal. It splits the data into buyer/seller perspectives, calculates individual Trade PnL (accounting for final prices, multipliers, and commissions), counts unique counterparties traded with, and produces a final blended rank.

### 5. [Rank Adjustment & Penalties (`Second_Last_Teams.py`)](https://www.google.com/search?q=%5Bhttps://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/Second_Last_Teams.py%5D(https://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/Second_Last_Teams.py))

**Purpose:** Manual Intervention & Disciplinary Scoring.
Allows administrators to manually apply massive NLV penalties (e.g., `-2,000,000`) to specific teams or non-participating traders. This effectively forces selected teams to the bottom of the rankings without deleting their data entirely. Generates adjusted CSV files for the next stage of the pipeline.

### 6. [NLV Aggregator (`NLV_calculator.py`)](https://www.google.com/search?q=%5Bhttps://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/NLV_calculator.py%5D(https://github.com/YiMingChangJ/RITC-Scoring-Extractor-and-Calculator/blob/main/RITCx%2520and%2520RITC%2520Scoring/NLV_calculator.py))

**Purpose:** Quick Data Consolidation.
A utility script that walks through a directory of raw CSV files, extracts the `TraderID` and `NLV`, derives the parent `TeamID`, and aggregates the total NLV per team into a single, clean CSV file.

---

## ⚙️ How the CaseRankAnalyzer Works (The P&L Pipeline)

The core scoring loop (`P&L_Ranking_Calculator`) is completely automated. It expects a main directory containing subfolders for each heat.

### 📂 Expected Folder Structure

```text
RITCx_Competition_Results/
│
├── ETF_Heat1/ 
│   └── Results.xlsx
├── ETF_Heat2/ 
│   └── Results.xlsx
├── Volatility_Heat1/ 
│   └── Results.xlsx
└── Volatility_Heat2/ 
    └── Results.xlsx

```

*Note: Each `Results.xlsx` file must contain team-level results (with `TraderID`, `NLV`, etc.).*

### 🔄 Processing Pipeline

1. **`load_and_prepare()`**
* Recursively searches all subfolders for `Results.xlsx`.
* Extracts NLV and applies case/heat labels based strictly on folder names (to prevent mis-mapped columns).


2. **`build_table()`**
* Aggregates team/trader P&L.
* **Densifies the Grid:** Identifies non-participants, assigns an NLV of `0`, and applies the `-inf` ranking penalty.
* Ranks participants per heat and per case.


3. **`save(filename)`**
* Saves the final wide table as an Excel (`.xlsx`) file containing the complete leaderboard, Average Case Ranks, and Overall Competition Ranks. Handles Windows file locks automatically by generating timestamped backups if the target file is open.

---

## ▶️ Example Usage (CaseRankAnalyzer)

To generate a complete competition leaderboard, simply drop all `Results.xlsx` files into their respective case subfolders and run the script:

```python
if __name__ == "__main__":
    main_path = r"C:\...\RITCx_Practice_Session_Results"
    
    analyzer = CaseRankAnalyzer(main_path)
    analyzer.load_and_prepare()
    analyzer.build_table()
    analyzer.save("Final_Team_Results.xlsx")

```
