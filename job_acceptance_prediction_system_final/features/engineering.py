"""features/engineering.py — Derived analytical features per project spec."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Experience Category (Fresher / Junior / Mid / Senior) ──
    raw_exp = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0)
    df["experience_category"] = pd.cut(
        raw_exp, bins=[-0.1,0,2,5,100],
        labels=["Fresher","Junior","Mid-Level","Senior"]).astype(str)

    # ── 2. Academic Performance Band ─────────────────────────────
    gpa_scaled = pd.to_numeric(df["gpa"], errors="coerce").fillna(7.0) * 10
    tenth      = pd.to_numeric(df["tenth_percentage"], errors="coerce").fillna(70)
    twelfth    = pd.to_numeric(df["twelfth_percentage"], errors="coerce").fillna(70)
    df["academic_score"] = (tenth*0.25 + twelfth*0.25 + gpa_scaled*0.50).round(2)
    df["academic_band"] = pd.cut(
        df["academic_score"], bins=[0,55,65,75,85,101],
        labels=["Poor","Below Avg","Average","Good","Excellent"]).astype(str)

    # ── 3. Skills Match Level ─────────────────────────────────────
    smp = pd.to_numeric(df["skills_match_pct"], errors="coerce").fillna(60)
    df["skills_level"] = pd.cut(smp, bins=[0,40,60,80,101],
                                 labels=["Low","Medium","High","Excellent"]).astype(str)

    # ── 4. Interview Performance Category ────────────────────────
    ints = pd.to_numeric(df["interview_score"], errors="coerce").fillna(60)
    df["interview_category"] = pd.cut(
        ints, bins=[0,40,55,70,85,101],
        labels=["Very Poor","Poor","Average","Good","Excellent"]).astype(str)

    # ── 5. Has Certification flag ─────────────────────────────────
    df["has_certification"] = (
        df["certification"].fillna("None").str.lower() != "none").astype(int)

    # ── 6. Placement Probability Score (0-100, business KPI) ─────
    int_n  = pd.to_numeric(df["interview_score"],     errors="coerce").fillna(50)
    skl_n  = pd.to_numeric(df["skills_match_pct"],    errors="coerce").fillna(50)
    emp_n  = pd.to_numeric(df["employability_score"], errors="coerce").fillna(50)
    apt_n  = pd.to_numeric(df["aptitude_score"],      errors="coerce").fillna(50)
    bl_n   = pd.to_numeric(df["backlogs"],            errors="coerce").fillna(0)

    df["placement_probability"] = np.clip(
        int_n*0.30 + skl_n*0.25 + df["academic_score"]*0.20
        + emp_n*0.15 + df["has_certification"]*10*0.10
        - bl_n*3, 0, 100).round(1)

    # ── 7. High Risk flag ─────────────────────────────────────────
    df["high_risk"] = (
        (df["placement_probability"] < 45) |
        (int_n < 40) | (bl_n >= 2)).astype(int)

    # ── 8. Salary Band ────────────────────────────────────────────
    sal = pd.to_numeric(df["offered_salary_inr"], errors="coerce").fillna(700000)
    df["salary_band"] = pd.cut(
        sal, bins=[0,500000,900000,1300000,3000000],
        labels=["<₹5L","₹5-9L","₹9-13L",">₹13L"]).astype(str)

    print(f"  ✓ Feature engineering — {df.shape[1]} total columns")
    return df
