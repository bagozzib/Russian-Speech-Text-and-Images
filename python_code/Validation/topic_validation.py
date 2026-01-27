# Python 3 (Google Colab)
!pip -q install gspread google-auth pandas numpy scikit-learn

import re, ast, io, os
import numpy as np
import pandas as pd
from datetime import datetime

import gspread
from google.colab import auth
from google.auth import default

from sklearn.metrics import precision_recall_fscore_support

# INPUTS: KREMLIN EN
MAIN_SHEET_URL  = "" #define your csv file or Google sheet link here
MAIN_GID        = ""

# Main columns
COL_ID_MAIN      = "id"
COL_DECL_TOPICS  = "declared_topics"

# OPTIONAL (only if exists; used for sanity checking, not needed for Ben)
COL_CURATED_TOPIC_ID = "curated_topic_id"   # optional

# Probs CSV (your Drive path)
PROBS_CSV_PATH = "" # give the long (N*K) Probability csv file url here

# Expected topics
K_TOPICS = 89  # topic_id 0..88

# MAPPING TABLE: topic_id -> declared_theme (+ confidence) This is for Kremlin English
MAPPING_CSV = """topic_id,topic_label,declared_theme,confidence
0,Russia's neighbours,CIS,high
1,Domestic coalitions,Parties,medium
2,Russia-West relations,Foreign policy,high
3,Crimea affairs,Foreign policy,high
4,WWII commemoration,Great Patriotic War,high
5,Putin's European allies,Foreign policy,medium
6,Intergovernmental cooperation,Foreign policy,medium
7,Russia-China relations,Foreign policy,high
8,Russian Far East,Regions,high
9,Domestic business,Support for business,high
10,Energy sector,Energy,high
11,Russia-Ukraine relations,Foreign policy,high
12,Olympics,Sport,high
13,Military and defense,Armed Forces,high
14,Domestic economy,Economy and finance,high
15,Healthcare system,Healthcare,high
16,Non-Western allies,Foreign policy,medium
17,Religion,Religion,high
18,National award ceremonies,State awards,high
19,East Asian relations,Foreign policy,medium
20,Education system,School,medium
21,Russia-ASEAN relations,Russia–ASEAN,high
22,Russia-Germany relations,Foreign policy,medium
23,Domestic law enforcement,Law enforcement agencies,high
24,Russia's navy,Armed Forces,medium
25,Russia-Eurasian cooperation,Foreign policy,medium
26,Security services,National security,high
27,Russia-European relations,Russia–European Union,medium
28,Domestic courts,Court,high
29,Russia-Georgia relations,Foreign policy,medium
30,Russia's transportation system,Transport,high
31,Russia-Africa relations,Foreign policy,medium
32,Domestic emergency response,National security,medium
33,Russia-India relations,Foreign policy,medium
34,Russia-Middle Eastern relations,Foreign policy,medium
35,Domestic financial institutions,Banks,high
36,Russia-Israel-Palestine relations,Foreign policy,medium
37,Russia's culture,Culture,high
38,Scientific developments,Science and innovation,high
39,Middle-Eastern partnership,Foreign policy,low
40,Russia-Iran alliance,Foreign policy,medium
41,Automotive industry,Industry,medium
42,Space industry,Space,high
43,Russia-Latin America relations,Foreign policy,medium
44,Chechnya affairs,National security,medium
45,Terrorism,Fight against terrorism,high
46,Aviation industry,Transport,medium
47,Russia-Bulgaria-Greece relations,Foreign policy,medium
48,Russia-Scandinavia relations,Foreign policy,medium
49,Domestic agriculture,Agriculture,high
50,Russia-Mongolia relations,Foreign policy,low
51,World Cup,Sport,high
52,Nuclear industry,Energy,medium
53,New Year's speeches,Culture,medium
54,Environmental protection,Environment,high
55,Elections,Parties,medium
56,Russia-Spain relations,Foreign policy,medium
57,Hockey,Sport,high
58,Russia-Egypt relations,Foreign policy,medium
59,Domestic volunteerism,Social services,medium
60,Construction industry,Construction industry,high
61,Domestic unions,Trade unions,high
62,Arctic exploration,Climate,medium
63,Emergency ministry meetings,National security,medium
64,Media,Media,high
65,Investments,Investment,high
66,Senior citizens,Pensions,medium
67,Financial monitoring,Economy and finance,medium
68,Child welfare policy,Children,high
69,Narcotics control,Fight against drugs,high
70,Family affairs,Demographics,medium
71,Russia-Brazil relations,Foreign policy,medium
72,Russia's youth,Demographics,medium
73,Border control,Migration,medium
74,Taxation,Taxes,high
75,Northwestern Europe affairs,Foreign policy,low
76,Women's recognition,Human rights,medium
77,Caspian region,Foreign policy,low
78,Russia-Afghanistan relations,Foreign policy,medium
79,Russia-UN relations,UN,high
80,Russia-Cuba relations,Foreign policy,medium
81,Russia-Indonesia relations,Foreign policy,medium
82,Russian Sports,Sport,high
83,Russia's geographical society,Environment,medium
84,Domestic tourism,Tourism,high
85,Students,School,medium
86,Agricultural industry,Agriculture,high
87,Russia-Cyprus relations,Foreign policy,medium
88,Customs service,Customs,high
"""

def parse_list_cell(cell):
    """Parse list or list-of-lists from sheet cell; flatten; drop 'more/Ещё N/Another' style summary tokens."""
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    if s in {"[]", "[ ]"}:
        return []

    try:
        v = ast.literal_eval(s)
    except Exception:
        v = [s]

    items = v if isinstance(v, list) else [v]

    # flatten
    flat = []
    for x in items:
        if isinstance(x, list):
            for y in x:
                t = str(y).strip()
                if t:
                    flat.append(t)
        else:
            t = str(x).strip()
            if t:
                flat.append(t)

    # remove summary tokens anywhere
    more_re   = re.compile(r"^\s*\d+\s+more\s*$", re.IGNORECASE)
    eshe_re   = re.compile(r"^\s*(Ещё|Еще)\s*\d+\s*$", re.IGNORECASE)
    another_re = re.compile(r"\banother\b", re.IGNORECASE)

    out = []
    for t in flat:
        if more_re.match(t) or eshe_re.match(t) or another_re.search(t):
            continue
        out.append(t)
    return out

def load_main_sheet_declared_only(gc):
    sh = gc.open_by_url(MAIN_SHEET_URL)
    ws = sh.get_worksheet_by_id(MAIN_GID)

    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        raise RuntimeError("Main sheet has no data rows.")

    header = vals[0]
    rows = vals[1:]
    df = pd.DataFrame(rows, columns=header)

    if COL_ID_MAIN not in df.columns or COL_DECL_TOPICS not in df.columns:
        raise RuntimeError(f"Main sheet missing required columns: {COL_ID_MAIN}, {COL_DECL_TOPICS}")

    df[COL_ID_MAIN] = pd.to_numeric(df[COL_ID_MAIN], errors="coerce").astype("Int64")
    df[COL_DECL_TOPICS] = df[COL_DECL_TOPICS].apply(parse_list_cell)

    # keep docs with non-empty declared topics
    df_decl = df[df[COL_DECL_TOPICS].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    df_decl = df_decl.dropna(subset=[COL_ID_MAIN]).copy()
    df_decl[COL_ID_MAIN] = df_decl[COL_ID_MAIN].astype(int)

    # optional curated topic id
    if COL_CURATED_TOPIC_ID in df_decl.columns:
        df_decl[COL_CURATED_TOPIC_ID] = pd.to_numeric(df_decl[COL_CURATED_TOPIC_ID], errors="coerce").astype("Int64")

    return df, df_decl

def load_probs_csv_long(path):
    dfp = pd.read_csv(path)

    # Your CSV columns:
    # ['ID', 'Topic ID', 'Topic Name', 'Topic Group Name', 'Probability score']
    cols = {c.lower(): c for c in dfp.columns}
    need = ["id", "topic id", "probability score"]
    for n in need:
        if n not in cols:
            raise RuntimeError(f"PROBS CSV missing column '{n}'. Found: {list(dfp.columns)}")

    dfp = dfp.rename(columns={
        cols["id"]: "id",
        cols["topic id"]: "topic_id",
        cols["probability score"]: "prob",
    })

    dfp["id"] = pd.to_numeric(dfp["id"], errors="coerce")
    dfp["topic_id"] = pd.to_numeric(dfp["topic_id"], errors="coerce")
    dfp["prob"] = pd.to_numeric(dfp["prob"], errors="coerce")

    dfp = dfp.dropna(subset=["id", "topic_id", "prob"]).copy()
    dfp["id"] = dfp["id"].astype(int)
    dfp["topic_id"] = dfp["topic_id"].astype(int)
    dfp["prob"] = dfp["prob"].astype(float)
    return dfp

def dominant_topic_per_doc(df_probs):
    # tie-break: higher prob first, then smaller topic_id
    df_sorted = df_probs.sort_values(["id", "prob", "topic_id"], ascending=[True, False, True])
    df_dom = df_sorted.groupby("id", as_index=False).first()[["id", "topic_id", "prob"]]
    df_dom = df_dom.rename(columns={"topic_id": "dom_topic_id", "prob": "dom_prob"})
    return df_dom

def build_mapping(map_df, conf_keep=None):
    m = map_df.copy()
    m["confidence"] = m["confidence"].astype(str).str.strip().str.lower()
    if conf_keep is not None:
        m = m[m["confidence"].isin(conf_keep)].copy()
    return dict(zip(m["topic_id"].astype(int).tolist(),
                    m["declared_theme"].astype(str).str.strip().tolist()))

def dominant_validation(df_decl, df_dom, mapping_dict, label_set):
    # align to declared subset
    df = df_decl[[COL_ID_MAIN, COL_DECL_TOPICS]].merge(df_dom, left_on=COL_ID_MAIN, right_on="id", how="inner")
    df["pred_theme"] = df["dom_topic_id"].map(mapping_dict)
    df["is_mapped"] = df["pred_theme"].notna()

    # Top-1 hit: predicted theme is in declared topics list
    def hit(row):
        th = row["pred_theme"]
        if pd.isna(th):
            return 0
        return 1 if th in row[COL_DECL_TOPICS] else 0

    df["top1_hit"] = df.apply(hit, axis=1)
    df_m = df[df["is_mapped"]].copy()

    if len(df_m) == 0:
        return df, {"n_docs_declared_eval": len(df), "n_docs_mapped_pred": 0}

    used_labels = label_set[:]  # paper-friendly (same eval labels across mapping sets)
    lab2i = {lab:i for i,lab in enumerate(used_labels)}
    Y_true = np.zeros((len(df_m), len(used_labels)), dtype=int)
    Y_pred = np.zeros((len(df_m), len(used_labels)), dtype=int)

    for r, (_, row) in enumerate(df_m.iterrows()):
        for t in row[COL_DECL_TOPICS]:
            if t in lab2i:
                Y_true[r, lab2i[t]] = 1
        pt = row["pred_theme"]
        if pt in lab2i:
            Y_pred[r, lab2i[pt]] = 1

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )

    truth_labels = sorted({t for lst in df[COL_DECL_TOPICS].tolist() for t in lst})
    metrics = {
        "n_docs_declared_eval": int(len(df)),
        "n_docs_mapped_pred": int(len(df_m)),
        "top1_hit_rate_all_docs": float(df["top1_hit"].mean()),
        "top1_hit_rate_mapped_docs": float(df_m["top1_hit"].mean()),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "n_labels_eval": int(len(used_labels)),
        "n_truth_labels_present": int(len(truth_labels)),
    }
    return df, metrics

# ---------- Google Sheets writing (chunked, safe) ----------
def write_df_to_ws(ws, df, chunk_rows=2000):
    df2 = df.copy()

    # convert list cells to strings so Sheets doesn't choke
    for c in df2.columns:
        df2[c] = df2[c].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

    df2 = df2.replace({np.nan: ""})

    values = [df2.columns.tolist()] + df2.astype(object).values.tolist()
    nrows = len(values)
    ncols = len(values[0]) if values else 1

    ws.resize(rows=nrows, cols=ncols)

    # write in chunks to avoid request size limits
    # chunk includes header only in first batch
    start = 0
    while start < nrows:
        end = min(start + chunk_rows, nrows)
        block = values[start:end]
        # A1 for header block, then A{row} for others
        a1 = f"A{start+1}"
        ws.update(range_name=a1, values=block)
        start = end

# RUN
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

print("Loading main sheet...")
df_main_all, df_main_decl = load_main_sheet_declared_only(gc)
print("  total docs:", df_main_all[COL_ID_MAIN].nunique())
print("  docs with declared_topics:", df_main_decl[COL_ID_MAIN].nunique())

print("\nReading probs CSV:", PROBS_CSV_PATH)
df_probs = load_probs_csv_long(PROBS_CSV_PATH)
print("  rows:", len(df_probs), " unique docs:", df_probs["id"].nunique())

# dominant topic per doc
df_dom = dominant_topic_per_doc(df_probs)

# OPTIONAL sanity: compare with curated_topic_id if present
curated_check_df = None
if COL_CURATED_TOPIC_ID in df_main_decl.columns:
    tmp = df_main_decl[[COL_ID_MAIN, COL_CURATED_TOPIC_ID]].dropna()
    tmp = tmp.merge(df_dom, left_on=COL_ID_MAIN, right_on="id", how="inner")
    tmp["match"] = (tmp[COL_CURATED_TOPIC_ID].astype(int) == tmp["dom_topic_id"].astype(int))
    curated_check_df = tmp.copy()
    print("\nOptional curated_topic_id check:")
    print("  compared docs:", len(tmp))
    print("  mismatches   :", int((~tmp["match"]).sum()))

# mapping df
map_df = pd.read_csv(io.StringIO(MAPPING_CSV))
map_df["topic_id"] = map_df["topic_id"].astype(int)
map_df["declared_theme"] = map_df["declared_theme"].astype(str).str.strip()
map_df["confidence"] = map_df["confidence"].astype(str).str.strip().str.lower()

# declared theme universe (truth)
declared_universe = sorted({t for lst in df_main_decl[COL_DECL_TOPICS].tolist() for t in lst})

# run 3 mapping sets
mapping_sets = [
    ("all", None),
    ("high+medium", {"high","medium"}),
    ("high", {"high"}),
]

rows_metrics = []
details_by_set = {}

for name, conf_keep in mapping_sets:
    mp = build_mapping(map_df, conf_keep=conf_keep)
    df_detail, metrics = dominant_validation(df_main_decl, df_dom, mp, label_set=declared_universe)
    metrics["mapping_set"] = name
    rows_metrics.append(metrics)
    details_by_set[name] = df_detail

df_metrics = pd.DataFrame(rows_metrics)

print("\n=== DOMINANT TOPIC VALIDATION METRICS ===")
print(df_metrics[[
    "mapping_set",
    "n_docs_declared_eval",
    "n_docs_mapped_pred",
    "top1_hit_rate_all_docs",
    "top1_hit_rate_mapped_docs",
    "precision_micro","recall_micro","f1_micro",
    "precision_macro","recall_macro","f1_macro"
]])

# WRITE OUTPUTS TO A NEW GOOGLE SHEET + PRINT LINKS
ts = datetime.now().strftime("%Y%m%d_%H%M")
out_title = f"Kremlin_EN_DominantTopicValidation_{ts}"
out_sh = gc.create(out_title)

# Tabs:
ws_metrics = out_sh.sheet1
ws_metrics.update_title("dominant_metrics")

ws_details_all = out_sh.add_worksheet(title="details_all_mapping", rows=1000, cols=10)
ws_mapping = out_sh.add_worksheet(title="mapping_used", rows=200, cols=10)

# Optional tab for curated sanity (only if present)
ws_curated = None
if curated_check_df is not None:
    ws_curated = out_sh.add_worksheet(title="curated_topic_sanity", rows=1000, cols=10)

# Write data
write_df_to_ws(ws_metrics, df_metrics, chunk_rows=2000)
write_df_to_ws(ws_details_all, details_by_set["all"], chunk_rows=2000)
write_df_to_ws(ws_mapping, map_df, chunk_rows=2000)

if curated_check_df is not None:
    write_df_to_ws(ws_curated, curated_check_df, chunk_rows=2000)

print("\n Output Google Sheet created:")
print(out_sh.url)
print("\nDirect tab links:")
print("dominant_metrics       :", f"{out_sh.url}#gid={ws_metrics.id}")
print("details_all_mapping    :", f"{out_sh.url}#gid={ws_details_all.id}")
print("mapping_used           :", f"{out_sh.url}#gid={ws_mapping.id}")
if ws_curated is not None:
    print("curated_topic_sanity   :", f"{out_sh.url}#gid={ws_curated.id}")
