import math
import numpy as np
import pandas as pd

SHEETS = {
    "Kremlin_RU": "https://docs.google.com/spreadsheets/d/1wvrmd7f59RrQn8ZL5KnMy1m8WmKSiceOgcFymNPRUjA/edit#gid=0",
    "Kremlin_EN": "https://docs.google.com/spreadsheets/d/1HETJopC1Ca7fd32-9eK_UsdmWNj8z0s-njTBWjXOTA0/edit#gid=0",
    "MID_RU":    "https://docs.google.com/spreadsheets/d/1e2x-ckyj4hoh-UaWeW_beYHhwfw2BCEkw_uMXTWLgFk/edit#gid=0",
    "MID_EN":    "https://docs.google.com/spreadsheets/d/1x5EEeRXLUAVC0k7BuTMwxENioDO9oOLtufnnHPjwbLA/edit#gid=0",
}

THRESHOLDS_KM = [5, 10, 50, 100]
MATCH_THRESHOLD_KM = 100

# Required input columns in each sheet
MACHINE_LAT = "machine_latitude"
MACHINE_LON = "machine_longitude"
MANUAL_LAT  = "manual_latitude"
MANUAL_LON  = "manual_longitude"

# Output columns we will create/fill
OUT_DISTANCE = "distance_km"
OUT_BUCKET   = "distance_bucket"
OUT_MATCH    = f"match_{MATCH_THRESHOLD_KM}km"


def get_gspread_client_colab():
    from google.colab import auth
    auth.authenticate_user()
    import gspread
    import google.auth
    creds, _ = google.auth.default()
    return gspread.authorize(creds)

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance (km).
    Inputs can be numpy arrays; returns numpy array.
    """
    R = 6371.0088  # mean Earth radius in km
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def to_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def make_bucket(dist_km):
    """
    dist_km: float or NaN
    Returns bucket label string.
    """
    if pd.isna(dist_km):
        return "MISSING_COORDS"
    for t in THRESHOLDS_KM:
        if dist_km <= t:
            return f"<= {t} km"
    return f"> {max(THRESHOLDS_KM)} km"

def make_match(dist_km):
    if pd.isna(dist_km):
        return "MISSING_COORDS"
    return "MATCH" if dist_km <= MATCH_THRESHOLD_KM else "NO_MATCH"

def ensure_columns(ws, required_cols):
    """
    Ensures header row contains required_cols (adds them at end if missing).
    Returns updated header list.
    """
    header = ws.row_values(1)
    header_set = set(header)

    missing = [c for c in required_cols if c not in header_set]
    if not missing:
        return header

    # Add missing columns at the end
    new_header = header + missing

    # If sheet doesn't have enough columns, extend it
    current_cols = len(header)
    needed_cols = len(new_header)
    if needed_cols > current_cols:
        ws.add_cols(needed_cols - current_cols)

    # Write updated header row
    ws.update(range_name=f"1:1", values=[new_header])
    return new_header

def update_columns(ws, header, df, col_names):
    """
    Updates specific columns in-place (same row order as sheet).
    Writes values from df[col] into the correct column positions.
    Assumes df has same number of rows as the sheet data (excluding header).
    """
    n = len(df)
    for col in col_names:
        if col not in header:
            raise ValueError(f"Column '{col}' not found in header after ensure_columns().")

        col_idx = header.index(col) + 1  # 1-based for Sheets
        # Prepare values as a column vector
        values = df[col].astype(str).replace({"nan": ""}).tolist()
        
        rng = f"{gspread.utils.rowcol_to_a1(2, col_idx)}:{gspread.utils.rowcol_to_a1(n+1, col_idx)}"
        ws.update(rng, [[v] for v in values])


def main_update_sheets():
    import gspread  # used by update_columns for utils
    gc = get_gspread_client_colab()

    for corpus, url in SHEETS.items():
        sh = gc.open_by_url(url)
        ws = sh.get_worksheet(0)  # gid=0

        records = ws.get_all_records()  # list of dicts
        df = pd.DataFrame(records)
        if df.empty:
            print(f"[{corpus}] Sheet is empty, skipping.")
            continue

        # Make sure required inputs exist
        for c in [MACHINE_LAT, MACHINE_LON, MANUAL_LAT, MANUAL_LON]:
            if c not in df.columns:
                raise ValueError(f"[{corpus}] Missing required column '{c}'. Found: {list(df.columns)}")

        # Compute distance
        mlats = to_float_series(df[MACHINE_LAT])
        mlons = to_float_series(df[MACHINE_LON])
        hlats = to_float_series(df[MANUAL_LAT])
        hlons = to_float_series(df[MANUAL_LON])

        both = mlats.notna() & mlons.notna() & hlats.notna() & hlons.notna()
        dist = pd.Series(np.nan, index=df.index, dtype="float64")
        if both.any():
            dist.loc[both] = haversine_km(mlats.loc[both].to_numpy(),
                                          mlons.loc[both].to_numpy(),
                                          hlats.loc[both].to_numpy(),
                                          hlons.loc[both].to_numpy())

        df[OUT_DISTANCE] = dist.round(6)  # keep stable precision
        df[OUT_BUCKET]   = df[OUT_DISTANCE].apply(make_bucket)
        df[OUT_MATCH]    = df[OUT_DISTANCE].apply(make_match)

        # Ensure output columns exist, then write back
        header = ensure_columns(ws, [OUT_DISTANCE, OUT_BUCKET, OUT_MATCH])
        header = ws.row_values(1)

        # Write only the three columns back
        # We must use gspread.utils, so import gspread inside this scope
        import gspread
        update_columns(ws, header, df, [OUT_DISTANCE, OUT_BUCKET, OUT_MATCH])

        print(f"[{corpus}] Updated {len(df)} rows with {OUT_DISTANCE}, {OUT_BUCKET}, {OUT_MATCH}.")

if __name__ == "__main__":
    main_update_sheets()
