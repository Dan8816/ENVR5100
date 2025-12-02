"""
NPDES Monitoring Data Analysis for City of Lawton WWTP (OK0035246)
=================================================================

This script supports the analysis described in Appendix A of the capstone:

- Reads EPA NPDES monitoring data from an Excel workbook:
  "Lawton NPDESMonitoringData_OK0035246.xlsx"

- Focuses on the sheet "Outfall 003 to Plant", which represents
  effluent from the Lawton WWTP to the power generation facility.

- Cleans header rows, reshapes data, and filters parameters of interest.

- Provides a framework for comparing measured values against permit limits
  (where available) and flagging exceedances.

Author: Danny Engle
"""

import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------------------------

# Path to the Excel file (adjust as needed)
DATA_DIR = pathlib.Path(".")
EXCEL_PATH = DATA_DIR / "Lawton NPDESMonitoringData_OK0035246.xlsx"

# Name of the sheet that contains Outfall 003 data
OUTFALL_SHEET_NAME = "Outfall 003 to Plant"

# Parameters of interest for this project
PARAMETERS_OF_INTEREST = [
    "CBOD5",          # Carbonaceous biochemical oxygen demand
    "CBOD",           # Alternate label sometimes used
    "TSS",            # Total suspended solids
    "AMMONIA",        # Ammonia as N
    "NH3",            # Alternate label
    "FECAL",          # Fecal coliform
    "E. COLI",
    "PH",             # pH
    "ZINC",           # Zinc (Zn)
    "COPPER",         # Copper (Cu)
]

# Optional: Manual dictionary of permit limits (example only; values and units
# should be checked against the actual OPDES/NPDES permit, not assumed here).
# Keys should match the normalized parameter_name column (see normalize_param())
PERMIT_LIMITS: Dict[str, Dict[str, float]] = {
    "CBOD5": {"mo_avg_mg_L": 1501.2, "daily_max_mg_L": 15.0},
    "TSS":   {"mo_avg_mg_L": 2251.8, "daily_max_mg_L": 22.5},
}


# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------

def normalize_param(raw_name: str) -> str:
    """
    Normalize a raw parameter column name to a simpler identifier.

    Example:
    "CBOD5, cBOD, mg/L" -> "CBOD5"
    "Total Suspended Solids, mg/L" -> "TSS"
    """
    if not isinstance(raw_name, str):
        return ""

    name = raw_name.upper()

    if "CBOD" in name:
        return "CBOD5"
    if "SUSPENDED SOLIDS" in name or "TSS" in name:
        return "TSS"
    if "AMMONIA" in name or "NH3" in name:
        return "AMMONIA_N"
    if "FECAL" in name or "E. COLI" in name:
        return "FECAL_COLIFORM"
    if "ZINC" in name:
        return "ZINC"
    if "COPPER" in name:
        return "COPPER"
    if name.strip() == "PH":
        return "PH"

    # Fallback to the first word
    return name.split(",")[0].strip()


def infer_header_row(df_raw: pd.DataFrame) -> int:
    """
    Infer the header row index for parameter columns.

    For the EPA NPDES Excel format used in this project, the "Outfall 003 to Plant"
    sheet typically has parameter names on a single row (e.g., row 3) and the
    monitoring period label "Mon Pd End Date:" somewhere in column 0 nearby.

    Here we:
    - Look for the row index containing "Mon Pd End Date" in the first column.
    - Assume the parameter names are 2 rows above that.
    - This assumption is based on manual inspection of the downloaded file.
    """

    col0 = df_raw.iloc[:, 0].astype(str).str.upper()
    candidates = df_raw.index[col0.str.contains("MON PD END DATE", na=False)]
    if len(candidates) == 0:
        raise ValueError("Could not find 'Mon Pd End Date' row in column 0.")
    mon_pd_row = candidates[0]

    # Based on inspection of the EPA file, parameter names are typically 2 rows above
    header_row = mon_pd_row - 2
    if header_row < 0:
        raise ValueError("Inferred header row index is negative; check file structure.")

    return header_row


def load_outfall_data(
    excel_path: pathlib.Path,
    sheet_name: str = OUTFALL_SHEET_NAME,
) -> pd.DataFrame:
    """
    Load and clean the Outfall 003 monitoring data from the Excel workbook.

    Returns a DataFrame in "wide" format, with:
    - A "Monitoring Period End Date" column
    - One column per parameter / statistic combination.
    """

    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # Infer which row contains parameter names
    header_row = infer_header_row(df_raw)

    # Use that row as the header
    df = df_raw.copy()
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1 :].reset_index(drop=True)

    # Rename the date column to something convenient
    # In the EPA file, this is usually labeled "Mon Pd End Date:" or similar.
    date_col_candidates = [
        c for c in df.columns if isinstance(c, str) and "MON PD END DATE" in c.upper()
    ]
    if len(date_col_candidates) == 0:
        # Fallback to the first column
        date_col = df.columns[0]
    else:
        date_col = date_col_candidates[0]

    df = df.rename(columns={date_col: "monitoring_period_end"})

    # Drop rows with no date (non-data rows at bottom or gaps)
    df = df[~df["monitoring_period_end"].isna()].copy()
    df["monitoring_period_end"] = pd.to_datetime(df["monitoring_period_end"])

    return df


def reshape_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide-format DataFrame to a long/tidy format:

    Columns:
    - monitoring_period_end
    - raw_parameter_col : original Excel column header (string)
    - parameter_name    : normalized name (e.g., 'CBOD5')
    - value             : numeric value (if convertible)

    We keep all columns except the date and collapse them with melt().
    """

    id_vars = ["monitoring_period_end"]
    value_vars = [c for c in df_wide.columns if c not in id_vars]

    df_long = df_wide.melt(
        id_vars=id_vars, value_vars=value_vars,
        var_name="raw_parameter_col", value_name="raw_value"
    )

    # Drop rows with no data
    df_long = df_long[~df_long["raw_value"].isna()].copy()

    # Normalize parameter names
    df_long["parameter_name"] = df_long["raw_parameter_col"].apply(normalize_param)

    # Convert values to numeric where possible; non-numeric become NaN
    df_long["value"] = pd.to_numeric(df_long["raw_value"], errors="coerce")

    # Keep rows with recognized numeric values
    df_long = df_long[~df_long["value"].isna()].copy()

    return df_long


def filter_parameters(df_long: pd.DataFrame, param_list: List[str]) -> pd.DataFrame:
    """
    Keep only the parameters of interest (based on substring matches against the
    normalized parameter_name).
    """

    # To make matching easier, just check if any of the tokens appear in the normalized name
    param_list_upper = [p.upper() for p in param_list]
    mask = df_long["parameter_name"].apply(
        lambda s: any(token in s.upper() for token in param_list_upper)
    )
    return df_long[mask].copy()


def flag_exceedances(df_params: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'exceedance' flag based on PERMIT_LIMITS.

    This is a simplified example. In practice, you would:
    - Distinguish between monthly averages vs daily maxima
    - Match the units (e.g., mg/L vs counts/100 mL)
    - Align the statistic (e.g., MO AVG vs DAILY MAX)

    Here we simply demonstrate the structure:
    - If a limit is defined in PERMIT_LIMITS for a parameter, we compare to a
      chosen limit (e.g., 'mo_avg_mg_L').
    """

    df = df_params.copy()
    df["exceedance"] = False

    for param, limits in PERMIT_LIMITS.items():
        if "mo_avg_mg_L" not in limits:
            continue
        limit = limits["mo_avg_mg_L"]

        mask_param = df["parameter_name"].str.upper() == param.upper()
        # Mark exceedance if value > limit
        df.loc[mask_param, "exceedance"] = df.loc[mask_param, "value"] > limit

    return df


def summarize_exceedances(df_params: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the number of observations and exceedances by parameter and year.
    """

    df = df_params.copy()
    df["year"] = df["monitoring_period_end"].dt.year

    summary = (
        df.groupby(["parameter_name", "year"])
        .agg(
            n_obs=("value", "count"),
            n_exceedances=("exceedance", "sum"),
        )
        .reset_index()
        .sort_values(["parameter_name", "year"])
    )

    return summary


# ---------------------------------------------------------------------------
# 3. Main analysis routine
# ---------------------------------------------------------------------------

def main():
    print("Loading Outfall 003 monitoring data from Excel...")
    df_wide = load_outfall_data(EXCEL_PATH, sheet_name=OUTFALL_SHEET_NAME)

    print(f"Wide-format data shape: {df_wide.shape}")

    print("Reshaping to long format...")
    df_long = reshape_to_long(df_wide)
    print(f"Long-format data shape: {df_long.shape}")

    print("Filtering to parameters of interest...")
    df_params = filter_parameters(df_long, PARAMETERS_OF_INTEREST)
    print(f"Filtered data shape (parameters of interest): {df_params.shape}")

    # Optional: apply exceedance logic if PERMIT_LIMITS has been populated
    if PERMIT_LIMITS:
        print("Flagging exceedances based on PERMIT_LIMITS...")
        df_params = flag_exceedances(df_params)
    else:
        # If limits are not specified, still add the column for consistency
        df_params["exceedance"] = np.nan

    # save the cleaned, long-format parameter data to CSV
    output_params_path = DATA_DIR / "lawton_outfall003_parameters_long.csv"
    df_params.to_csv(output_params_path, index=False)
    print(f"Saved long-format parameter data to: {output_params_path}")

    # If we have exceedance information, summarize by parameter and year
    if "exceedance" in df_params.columns:
        summary = summarize_exceedances(df_params)
        output_summary_path = DATA_DIR / "lawton_outfall003_exceedance_summary.csv"
        summary.to_csv(output_summary_path, index=False)
        print(f"Saved exceedance summary to: {output_summary_path}")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
