"""
preprocessing/pipeline.py
--------------------------
Full data cleaning & preprocessing pipeline:
  Step 1 — Remove duplicates
  Step 2 — Standardize inconsistent categoricals
  Step 3 — Handle missing values (median/mode)
  Step 4 — Logical consistency checks
  Step 5 — Encode target variable
  Step 6 — Label-encode categoricals + scale numerics
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.settings import OUT_DIR, RPT_DIR, TARGET

# ── Canonical maps ────────────────────────────────────────────────
TIER_MAP  = {k: v for d in [
    {a:"Tier 1" for a in ["tier1","tier 1","tier-1","t1","TIER1","TIER 1","Tier-1"]},
    {a:"Tier 2" for a in ["tier2","tier 2","tier-2","t2","TIER2","TIER 2","Tier-2"]},
    {a:"Tier 3" for a in ["tier3","tier 3","tier-3","t3","TIER3","TIER 3","Tier-3"]},
] for k,v in d.items()}

COMP_MAP  = {k: v for d in [
    {a:"Low"    for a in ["low","LOW","l","L"]},
    {a:"Medium" for a in ["medium","MEDIUM","med","Med"]},
    {a:"High"   for a in ["high","HIGH","h","H"]},
] for k,v in d.items()}

GEN_MAP   = {k: v for d in [
    {a:"Male"   for a in ["male","MALE","m","M"]},
    {a:"Female" for a in ["female","FEMALE","f","F"]},
    {a:"Other"  for a in ["other","OTHER"]},
] for k,v in d.items()}

CAT_MAPS  = {"company_tier": TIER_MAP,
             "competition_level": COMP_MAP,
             "gender": GEN_MAP}

NUM_FEATURES = ["tenth_percentage","twelfth_percentage","gpa","backlogs",
                "years_experience","num_skills","skills_match_pct",
                "projects_done","num_interviews","interview_score",
                "aptitude_score","employability_score","offered_salary_inr",
                "location_pref_match","age"]

CAT_FEATURES = ["gender","degree","certification","company_tier",
                "job_role","competition_level"]


def step1_remove_duplicates(df):
    before = len(df)
    key = [c for c in ["tenth_percentage","twelfth_percentage","gpa",
                        "interview_score","company_tier","status"] if c in df.columns]
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    return df, {"before": before, "removed": before-len(df), "after": len(df)}


def step2_standardize_cats(df):
    changed = {}
    for col, cmap in CAT_MAPS.items():
        if col not in df.columns: continue
        before = df[col].copy()
        df[col] = df[col].apply(
            lambda x: cmap.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
        changed[col] = int((df[col].fillna("") != before.fillna("")).sum())
    return df, {"changed": changed}


def step3_handle_missing(df):
    report = {}
    for col in NUM_FEATURES:
        if col not in df.columns: continue
        n = int(df[col].isnull().sum())
        if n:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            report[col] = {"method":"median","filled":n,"value":round(float(med),3)}
    for col in CAT_FEATURES:
        if col not in df.columns: continue
        n = int(df[col].isnull().sum())
        if n:
            mode = df[col].mode()[0] if df[col].notna().any() else "Unknown"
            df[col] = df[col].fillna(mode)
            report[col] = {"method":"mode","filled":n,"value":mode}
    return df, {"imputed": report,
                "total_filled": sum(v["filled"] for v in report.values())}


def step4_consistency(df):
    fixes = {}
    bounds = {"gpa":(4.0,10.0),"tenth_percentage":(0,100),"twelfth_percentage":(0,100),
              "interview_score":(0,100),"skills_match_pct":(0,100),
              "aptitude_score":(0,100),"employability_score":(0,100)}
    for col,(lo,hi) in bounds.items():
        if col not in df.columns: continue
        bad = (df[col]<lo)|(df[col]>hi)
        if bad.any():
            df.loc[bad, col] = df[col].median()
            fixes[col] = int(bad.sum())
    return df, {"fixes": fixes}


def step5_encode_target(df):
    df["status_num"] = df[TARGET].map({"Placed":1,"Not Placed":0}).astype(int)
    dist = df[TARGET].value_counts().to_dict()
    return df, {"distribution": dist,
                "placement_rate": round(df["status_num"].mean()*100, 2)}


def step6_encode_scale(df):
    """Label-encode categoricals and return fitted scaler (save both)."""
    encoders = {}
    for col in CAT_FEATURES:
        if col not in df.columns: continue
        le = LabelEncoder()
        df[col+"_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    num_present = [c for c in NUM_FEATURES if c in df.columns]
    df[num_present] = scaler.fit_transform(df[num_present].values)

    # Save artefacts
    joblib.dump(scaler,   OUT_DIR/"scaler.pkl")
    joblib.dump(encoders, OUT_DIR/"encoders.pkl")
    return df, scaler, encoders


def run_pipeline(df: pd.DataFrame, verbose=True) -> tuple:
    report = {}
    def log(msg):
        if verbose: print(msg)

    log("\n" + "═"*60)
    log("  🔧  DATA CLEANING & PREPROCESSING PIPELINE")
    log("═"*60)
    log(f"  Input : {len(df):,} rows × {df.shape[1]} cols  |  "
        f"Missing: {int(df.isnull().sum().sum()):,}")

    df, r = step1_remove_duplicates(df); report["S1_duplicates"] = r
    log(f"  ✓ S1 Duplicates   — removed {r['removed']:,}")

    df, r = step2_standardize_cats(df); report["S2_categoricals"] = r
    log(f"  ✓ S2 Categoricals — {r['changed']}")

    df, r = step3_handle_missing(df);  report["S3_missing"] = r
    log(f"  ✓ S3 Missing      — {r['total_filled']:,} cells imputed")

    df, r = step4_consistency(df);     report["S4_consistency"] = r
    log(f"  ✓ S4 Consistency  — {r['fixes']}")

    df, r = step5_encode_target(df);   report["S5_target"] = r
    log(f"  ✓ S5 Target       — {r['distribution']}  "
        f"| Placement rate: {r['placement_rate']}%")

    # Save cleaned CSV (before scaling for dashboard readability)
    clean_path = OUT_DIR / "job_placement_cleaned.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8")
    log(f"  ✓ Cleaned CSV → {clean_path}")

    df, scaler, encoders = step6_encode_scale(df)
    log(f"  ✓ S6 Encoding+Scaling done")
    log(f"\n  Output: {len(df):,} rows × {df.shape[1]} cols  |  "
        f"Missing: {int(df.isnull().sum().sum()):,}")
    log("═"*60 + "\n")

    # Save text report
    lines = ["JOB ACCEPTANCE — DATA CLEANING REPORT","="*55]
    for k,v in report.items():
        lines += [f"\n[{k}]"]
        for kk,vv in (v.items() if isinstance(v,dict) else [("value",v)]):
            lines.append(f"  {kk}: {vv}")
    (RPT_DIR/"cleaning_report.txt").write_text("\n".join(lines), encoding="utf-8")

    return df, report, scaler, encoders


if __name__ == "__main__":
    from data.generator import generate
    raw = generate(save=False)
    clean, rpt, *_ = run_pipeline(raw)
    print("Clean shape:", clean.shape)
