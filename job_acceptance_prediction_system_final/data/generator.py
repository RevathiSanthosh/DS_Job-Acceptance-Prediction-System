"""
data/generator.py
-----------------
Generates 50,000 realistic HR recruitment records with intentional
data quality issues: missing values, inconsistent categoricals,
duplicate-like records — exactly as specified in the GUVI document.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from config.settings import SEED, N_ROWS, DATA_DIR

rng  = np.random.default_rng(SEED)
np.random.seed(SEED)

JOB_ROLES   = ["Software Engineer","Data Analyst","ML Engineer","Product Manager",
                "Business Analyst","DevOps Engineer","Frontend Developer","Backend Developer",
                "Data Scientist","QA Engineer"]
CERTS       = ["AWS","Google Cloud","Azure","PMP","Salesforce",
                "Coursera ML","None","None","None","None"]
COMP_TIER   = ["Tier 1","Tier 2","Tier 3"]
COMP_LEVEL  = ["Low","Medium","High"]
GENDER      = ["Male","Female","Other"]
DEGREE      = ["B.Tech","BCA","BSc","MCA","MBA","B.Com"]

# ── dirty variants injected intentionally ────────────────────────
TIER_DIRTY  = {"Tier 1":["tier1","TIER1","tier 1","Tier-1","t1"],
               "Tier 2":["tier2","TIER2","tier 2","Tier-2","t2"],
               "Tier 3":["tier3","TIER3","tier 3","Tier-3","t3"]}
COMP_DIRTY  = {"Low":["low","LOW","L","l"],"Medium":["medium","MEDIUM","Med","med"],
               "High":["high","HIGH","H","h"]}
GEN_DIRTY   = {"Male":["male","MALE","M","m"],"Female":["female","FEMALE","F","f"]}


def _inject_null(arr, p=0.07):
    mask = rng.random(len(arr)) < p
    r = arr.astype(object); r[mask] = np.nan
    return r


def _dirty_cat(series, noise_map, p=0.12):
    vals = series.values.copy().astype(object)
    for i, v in enumerate(vals):
        if pd.notna(v) and rng.random() < p:
            opts = noise_map.get(str(v), [])
            if opts: vals[i] = rng.choice(opts)
    return vals


def generate(n: int = N_ROWS, save: bool = True) -> pd.DataFrame:
    # ── Simulate realistic correlations ──────────────────────────
    gpa          = np.clip(rng.normal(7.2, 1.3, n), 4.0, 10.0).round(2)
    tenth        = np.clip(rng.normal(75, 12, n), 40, 100).round(1)
    twelfth      = np.clip(rng.normal(73, 13, n), 40, 100).round(1)
    backlogs     = rng.integers(0, 5, n)
    yrs_exp      = np.clip(rng.exponential(2.3, n), 0, 15).round(1)
    num_skills   = rng.integers(1, 10, n)
    skill_match  = np.clip(rng.normal(61, 19, n), 5, 100).round(1)
    certs        = rng.choice(CERTS, n)
    projects     = rng.integers(0, 9, n)
    num_interv   = rng.integers(1, 7, n)
    int_score    = np.clip(rng.normal(63, 16, n), 20, 100).round(1)
    apt_score    = np.clip(rng.normal(57, 17, n), 10, 100).round(1)
    empl_score   = np.clip(rng.normal(59, 18, n), 10, 100).round(1)
    tier         = rng.choice(COMP_TIER, n, p=[0.20, 0.45, 0.35])
    job_role     = rng.choice(JOB_ROLES, n)
    salary       = np.where(tier=="Tier 1", rng.integers(1200000,2500001,n),
                   np.where(tier=="Tier 2", rng.integers(700000,1300001,n),
                             rng.integers(350000,750001,n)))
    comp_lvl     = rng.choice(COMP_LEVEL, n, p=[0.25,0.45,0.30])
    loc_match    = rng.choice([0,1], n, p=[0.30,0.70])
    age          = rng.integers(20, 36, n)
    gender       = rng.choice(GENDER, n, p=[0.57,0.40,0.03])
    degree       = rng.choice(DEGREE, n)

    # ── Logistic target with realistic feature weights ────────────
    z = (0.05*(gpa-7.0)
       + 0.030*(int_score-60)
       + 0.025*(skill_match-60)
       + 0.020*(empl_score-60)
       + 0.018*(apt_score-55)
       - 0.12*backlogs
       + 0.35*(tier=="Tier 1").astype(float)
       + 0.18*(tier=="Tier 2").astype(float)
       + 0.22*(certs != "None").astype(float)
       + 0.15*(comp_lvl=="Low").astype(float)
       - 0.12*(comp_lvl=="High").astype(float)
       + 0.12*loc_match
       + 0.015*yrs_exp
       + rng.normal(0, 0.35, n))
    prob   = 1 / (1 + np.exp(-z))
    status = np.where(rng.random(n) < prob, "Placed", "Not Placed")

    df = pd.DataFrame({
        "candidate_id"        : [f"CAND{i:06d}" for i in range(n)],
        "age"                 : age,
        "gender"              : gender,
        "degree"              : degree,
        "tenth_percentage"    : tenth,
        "twelfth_percentage"  : twelfth,
        "gpa"                 : gpa,
        "backlogs"            : backlogs,
        "years_experience"    : yrs_exp,
        "num_skills"          : num_skills,
        "skills_match_pct"    : skill_match,
        "certification"       : certs,
        "projects_done"       : projects,
        "num_interviews"      : num_interv,
        "interview_score"     : int_score,
        "aptitude_score"      : apt_score,
        "employability_score" : empl_score,
        "company_tier"        : tier,
        "job_role"            : job_role,
        "offered_salary_inr"  : salary,
        "competition_level"   : comp_lvl,
        "location_pref_match" : loc_match,
        "status"              : status,
    })

    # ── Inject data quality issues ────────────────────────────────
    for col, p in [("gpa",0.06),("interview_score",0.08),
                   ("skills_match_pct",0.07),("employability_score",0.09),
                   ("years_experience",0.05),("aptitude_score",0.10),
                   ("offered_salary_inr",0.04),("certification",0.06)]:
        df[col] = _inject_null(df[col].values, p)

    df["company_tier"]    = _dirty_cat(df["company_tier"],   TIER_DIRTY)
    df["competition_level"]= _dirty_cat(df["competition_level"], COMP_DIRTY)
    df["gender"]          = _dirty_cat(df["gender"],         GEN_DIRTY)

    # ── Inject ~2% duplicates ─────────────────────────────────────
    nd   = int(n * 0.02)
    dups = df.iloc[rng.integers(0, n, nd)].copy()
    df   = pd.concat([df, dups], ignore_index=True)
    df   = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if save:
        p = DATA_DIR / "job_placement_raw.csv"
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"  ✓ Raw dataset → {p}  ({len(df):,} rows × {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    df = generate()
    print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print(f"\nStatus dist:\n{df['status'].value_counts()}")
