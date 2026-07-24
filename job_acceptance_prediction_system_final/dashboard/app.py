"""
dashboard/app.py
-----------------
Job Acceptance Prediction System — Interactive Streamlit Dashboard
KPIs: Total Candidates, Placement Rate, Avg Interview Score,
      Avg Skills Match, Offer Dropout Rate, High-Risk %
Pages: Overview | EDA | Model Performance | Predict
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

from config.settings import OUT_DIR, MODEL_DIR, EDA_DIR, C, PLOTLY_COLORS

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  .stApp {{ background-color:{C['bg']}; }}
  .block-container {{ padding-top:0.8rem; padding-bottom:1rem; }}
  section[data-testid="stSidebar"] {{ background:{C['card']}; border-right:1px solid {C['border']}; }}

  .kpi-card {{
    background:linear-gradient(135deg,{C['card']},{C['bg']});
    border:1px solid {C['border']}; border-radius:14px;
    padding:18px 16px; text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.4);
  }}
  .kpi-val  {{ font-size:2rem; font-weight:800; margin:0; line-height:1.1; }}
  .kpi-lbl  {{ font-size:.7rem; color:{C['muted']}; margin-top:5px;
               letter-spacing:.7px; text-transform:uppercase; }}
  .kpi-delta{{ font-size:.8rem; margin-top:4px; }}

  .section-hdr {{
    border-left:4px solid {C['primary']}; padding-left:10px;
    color:{C['primary']}; font-weight:700; font-size:1rem; margin-bottom:6px;
  }}
  .stTabs [data-baseweb="tab"] {{ color:{C['muted']}; }}
  .stTabs [aria-selected="true"] {{
    color:{C['primary']} !important;
    border-bottom-color:{C['primary']} !important;
  }}
  div[data-testid="stMetric"] {{ background:{C['card']}; border-radius:10px; padding:12px; }}
</style>
""", unsafe_allow_html=True)

T = "plotly_dark"

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Loading pipeline…")
def load_data():
    from data.generator       import generate
    from preprocessing.pipeline import run_pipeline
    from features.engineering  import add_features

    csv_path = OUT_DIR / "job_placement_cleaned.csv"
    if csv_path.exists():
        raw = pd.read_csv(csv_path)
        # re-add derived features if missing
        if "placement_probability" not in raw.columns:
            raw = add_features(raw)
        return raw
    # Fresh run
    raw   = generate(save=True)
    clean, *_ = run_pipeline(raw, verbose=False)
    clean = add_features(clean)
    return clean

@st.cache_resource(show_spinner="🤖 Loading models…")
def load_models():
    try:
        from models.training import train_all
        from features.engineering import add_features
        csv_path = OUT_DIR / "job_placement_cleaned.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "placement_probability" not in df.columns:
                df = add_features(df)
        else:
            from data.generator import generate
            from preprocessing.pipeline import run_pipeline
            raw = generate(save=False)
            df, *_ = run_pipeline(raw, verbose=False)
            df = add_features(df)
        results = train_all(df)
        return results
    except Exception as e:
        st.warning(f"Model training skipped: {e}")
        return {}

df_all = load_data()

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
def sidebar(df):
    with st.sidebar:
        st.markdown(f"<h2 style='color:{C['primary']};margin:0'>🎯 Job Acceptance</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{C['muted']};font-size:.8rem'>Prediction System · GUVI HCL</p>", unsafe_allow_html=True)
        st.divider()

        # Filters
        st.markdown(f"<div style='color:{C['accent']};font-weight:700;font-size:.85rem'>🔽 FILTERS</div>", unsafe_allow_html=True)

        tiers = ["All"] + sorted(df["company_tier"].dropna().unique().tolist())
        sel_tier = st.selectbox("Company Tier", tiers)

        comp_lvls = ["All"] + sorted(df["competition_level"].dropna().unique().tolist())
        sel_comp  = st.selectbox("Competition Level", comp_lvls)

        if "experience_category" in df.columns:
            exp_cats = ["All"] + sorted(df["experience_category"].dropna().unique().tolist())
            sel_exp  = st.selectbox("Experience", exp_cats)
        else:
            sel_exp = "All"

        sel_status = st.radio("Status", ["All","Placed","Not Placed"])

        st.divider()
        pages = ["📊 Overview", "📈 EDA Charts", "🤖 Model Results", "🔮 Predict"]
        page  = st.radio("📄 Navigate", pages, label_visibility="collapsed")

        st.divider()
        st.markdown(f"<div style='font-size:.7rem;color:{C['muted']};text-align:center'>"
                    f"{len(df):,} records<br>GUVI HCL DS Program</div>", unsafe_allow_html=True)

    # Apply filters
    f = df.copy()
    if sel_tier   != "All": f = f[f["company_tier"]       == sel_tier]
    if sel_comp   != "All": f = f[f["competition_level"]  == sel_comp]
    if sel_exp    != "All" and "experience_category" in f.columns:
                             f = f[f["experience_category"]== sel_exp]
    if sel_status != "All": f = f[f["status"]             == sel_status]
    return f, page

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def kpi(col, val, label, color=None, delta=None):
    color = color or C["primary"]
    delta_html = f'<div class="kpi-delta" style="color:{color}">{delta}</div>' if delta else ""
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-val" style="color:{color}">{val}</div>
      <div class="kpi-lbl">{label}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def sh(title):
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)

def lay(fig, h=420):
    fig.update_layout(template=T, paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
                      height=h, margin=dict(l=10,r=10,t=40,b=10), font_color=C["text"])
    return fig

def pc(fig, **kw):
    st.plotly_chart(fig, use_container_width=True, **kw)

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
def page_overview(df):
    st.title("📊 Executive Overview")
    st.caption("Key Performance Indicators and high-level placement analytics")

    # ── KPIs ────────────────────────────────────────────────────────
    total      = len(df)
    placed_n   = (df["status"]=="Placed").sum()
    place_rate = placed_n/total*100 if total else 0
    dropout    = 100 - place_rate
    avg_int    = pd.to_numeric(df["interview_score"], errors="coerce").mean()
    avg_skill  = pd.to_numeric(df["skills_match_pct"], errors="coerce").mean()
    high_risk  = df["high_risk"].mean()*100 if "high_risk" in df.columns else 0
    avg_prob   = df["placement_probability"].mean() if "placement_probability" in df.columns else 0

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    kpi(c1, f"{total:,}",         "Total Candidates",      C["primary"])
    kpi(c2, f"{place_rate:.1f}%", "Placement Rate",        C["success"])
    kpi(c3, f"{dropout:.1f}%",    "Offer Dropout Rate",    C["danger"])
    kpi(c4, f"{avg_int:.1f}",     "Avg Interview Score",   C["accent"])
    kpi(c5, f"{avg_skill:.1f}%",  "Avg Skills Match",      C["info"])
    kpi(c6, f"{high_risk:.1f}%",  "High-Risk Candidates",  C["danger"])
    kpi(c7, f"{avg_prob:.1f}",    "Avg Placement Score",   C["primary"])

    st.markdown("---")

    c1, c2 = st.columns(2)

    # Placement donut
    with c1:
        sh("🥧 Placement Split")
        vc = df["status"].value_counts()
        fig = go.Figure(go.Pie(
            labels=vc.index, values=vc.values,
            hole=0.5, marker_colors=[C["success"],C["danger"]],
            textinfo="label+percent",
            marker_line=dict(color=C["bg"], width=2)))
        fig.update_layout(showlegend=True)
        pc(lay(fig, 380))

    # Placement by tier
    with c2:
        sh("🏢 Placement Rate by Company Tier")
        tr = df.groupby("company_tier")["status_num"].mean().reset_index()
        tr["rate"] = tr["status_num"]*100
        fig = px.bar(tr, x="company_tier", y="rate",
                     color="company_tier", color_discrete_sequence=PLOTLY_COLORS,
                     labels={"rate":"Placement Rate (%)","company_tier":"Tier"},
                     text=tr["rate"].round(1).astype(str)+"%")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_ticksuffix="%")
        pc(lay(fig, 380))

    # Monthly-style trend (by age as proxy)
    sh("📅 Placement Rate by Experience Category")
    if "experience_category" in df.columns:
        ec = df.groupby(["experience_category","status"]).size().reset_index(name="n")
        order = ["Fresher","Junior","Mid-Level","Senior"]
        ec["experience_category"] = pd.Categorical(ec["experience_category"], categories=order, ordered=True)
        ec = ec.sort_values("experience_category")
        fig = px.bar(ec, x="experience_category", y="n", color="status",
                     barmode="group",
                     color_discrete_map={"Placed":C["success"],"Not Placed":C["danger"]},
                     labels={"n":"Count","experience_category":"Experience","status":"Status"})
        pc(lay(fig, 380))

    c3, c4 = st.columns(2)
    with c3:
        sh("⚔️ Competition Level Impact")
        cl = df.groupby("competition_level")["status_num"].mean().reset_index()
        cl["rate"] = cl["status_num"]*100
        fig = px.bar(cl, x="competition_level", y="rate",
                     color="competition_level",
                     color_discrete_map={"Low":C["success"],"Medium":C["accent"],"High":C["danger"]},
                     labels={"rate":"Acceptance Rate (%)","competition_level":"Competition"},
                     text=cl["rate"].round(1).astype(str)+"%")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_ticksuffix="%")
        pc(lay(fig, 360))

    with c4:
        sh("💼 Top Job Roles by Application Volume")
        jr = df["job_role"].value_counts().head(8).reset_index()
        jr.columns = ["job_role","count"]
        fig = px.bar(jr, x="count", y="job_role", orientation="h",
                     color="count", color_continuous_scale="Purples",
                     labels={"count":"Candidates","job_role":"Role"})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        pc(lay(fig, 360))

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — EDA CHARTS
# ═══════════════════════════════════════════════════════════════════
def page_eda(df):
    st.title("📈 Exploratory Data Analysis")
    st.caption("Deep-dive into feature distributions, correlations and placement drivers")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎓 Academic","🎤 Interview & Skills","🏅 Certifications","🔥 Correlations"])

    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            sh("GPA Distribution by Outcome")
            fig = px.histogram(df, x="gpa", color="status",
                               color_discrete_map={"Placed":C["success"],"Not Placed":C["danger"]},
                               barmode="overlay", nbins=30, opacity=0.7,
                               labels={"gpa":"GPA","count":"Count"})
            pc(lay(fig, 360))
        with c2:
            sh("10th & 12th % by Outcome")
            fig = make_subplots(rows=1, cols=2, subplot_titles=["10th %","12th %"])
            for i, col in enumerate(["tenth_percentage","twelfth_percentage"],1):
                for st2, clr in [("Placed",C["success"]),("Not Placed",C["danger"])]:
                    vals = df[df["status"]==st2][col].dropna()
                    fig.add_trace(go.Histogram(x=vals, name=st2, marker_color=clr,
                                               opacity=0.7, showlegend=(i==1)), row=1, col=i)
            fig.update_layout(barmode="overlay")
            pc(lay(fig, 360))

        sh("Academic Band vs Placement Rate")
        if "academic_band" in df.columns:
            ab = df.groupby("academic_band")["status_num"].mean().reset_index()
            ab["rate"] = ab["status_num"]*100
            order = ["Poor","Below Avg","Average","Good","Excellent"]
            ab["academic_band"] = pd.Categorical(ab["academic_band"], categories=order, ordered=True)
            ab = ab.sort_values("academic_band")
            fig = px.bar(ab, x="academic_band", y="rate",
                         color="rate", color_continuous_scale="RdYlGn",
                         text=ab["rate"].round(1).astype(str)+"%",
                         labels={"rate":"Placement Rate %","academic_band":"Band"})
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, yaxis_ticksuffix="%")
            pc(lay(fig, 360))

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            sh("Interview Score Distribution")
            fig = px.histogram(df, x="interview_score", color="status",
                               color_discrete_map={"Placed":C["success"],"Not Placed":C["danger"]},
                               barmode="overlay", nbins=30, opacity=0.7)
            pc(lay(fig, 360))
        with c2:
            sh("Employability Score Distribution")
            fig = px.histogram(df, x="employability_score", color="status",
                               color_discrete_map={"Placed":C["success"],"Not Placed":C["danger"]},
                               barmode="overlay", nbins=30, opacity=0.7)
            pc(lay(fig, 360))

        sh("Skills Match % — Box Plot by Outcome")
        fig = px.box(df, x="status", y="skills_match_pct", color="status",
                     color_discrete_map={"Placed":C["success"],"Not Placed":C["danger"]},
                     notched=True, points="outliers",
                     labels={"skills_match_pct":"Skills Match %","status":"Outcome"})
        fig.update_layout(showlegend=False)
        pc(lay(fig, 380))

        sh("Aptitude Score vs Interview Score — Scatter")
        sample = df.sample(min(3000,len(df)), random_state=42)
        fig = px.scatter(sample, x="aptitude_score", y="interview_score",
                         color="status",
                         color_discrete_map={"Placed":C["success"],"Not Placed":C["danger"]},
                         opacity=0.5,
                         labels={"aptitude_score":"Aptitude Score","interview_score":"Interview Score"})
        pc(lay(fig, 420))

    with tab3:
        c1,c2 = st.columns(2)
        with c1:
            sh("Acceptance Rate by Certification")
            cr = df.groupby("certification")["status_num"].mean().reset_index()
            cr["rate"] = cr["status_num"]*100
            cr = cr.sort_values("rate", ascending=False)
            fig = px.bar(cr, x="rate", y="certification", orientation="h",
                         color="rate", color_continuous_scale="Plasma",
                         labels={"rate":"Acceptance Rate %","certification":"Cert"})
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            pc(lay(fig, 420))
        with c2:
            sh("Certified vs Not Certified")
            if "has_certification" in df.columns:
                hc = df.groupby("has_certification")["status_num"].mean().reset_index()
                hc["label"] = hc["has_certification"].map({0:"No Cert",1:"Has Cert"})
                hc["rate"]  = hc["status_num"]*100
                fig = px.bar(hc, x="label", y="rate",
                             color="label",
                             color_discrete_sequence=[C["danger"],C["success"]],
                             text=hc["rate"].round(1).astype(str)+"%",
                             labels={"rate":"Placement Rate %","label":""})
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, yaxis_ticksuffix="%")
                pc(lay(fig, 420))

    with tab4:
        sh("Feature Correlation Matrix")
        num_cols = ["gpa","tenth_percentage","twelfth_percentage","years_experience",
                    "skills_match_pct","interview_score","employability_score",
                    "aptitude_score","backlogs","status_num","placement_probability"]
        num_cols = [c for c in num_cols if c in df.columns]
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r",
                        labels=dict(color="Correlation"))
        fig.update_traces(textfont_size=9)
        pc(lay(fig, 520))

        sh("Placement Probability Distribution (Violin)")
        if "placement_probability" in df.columns:
            fig = px.violin(df, y="placement_probability", x="status",
                            color="status", box=True, points=False,
                            color_discrete_map={"Placed":C["success"],"Not Placed":C["danger"]})
            fig.update_layout(showlegend=False)
            pc(lay(fig, 380))

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════
def page_models():
    st.title("🤖 Model Performance")
    st.caption("Comparison of 6 ML algorithms — best model selected by AUC-ROC")

    # Check for saved chart images
    charts = {
        "16_roc_curves.png"      : "ROC Curves — All 6 Models",
        "17_model_comparison.png" : "Model Metrics Comparison",
        "18_feature_importance.png":"Feature Importance (Best Model)",
        "19_confusion_matrix.png" : "Confusion Matrix (Best Model)",
        "20_algorithm_ranking.png": "Algorithm Ranking by AUC-ROC",
    }

    for fname, title in charts.items():
        path = EDA_DIR / fname
        if path.exists():
            sh(title)
            st.image(str(path), use_column_width=True)
            st.markdown("")
        else:
            st.info(f"Run `python setup.py` to generate: {fname}")

    # Summary table
    sh("📋 Model Comparison Summary")
    report_path = Path(__file__).parent.parent / "outputs" / "reports" / "model_evaluation.txt"
    if report_path.exists():
        txt = report_path.read_text()
        # Parse metrics
        rows = []
        current = {}
        for line in txt.split("\n"):
            if line.startswith("Model:"):
                if current: rows.append(current)
                current = {"Model": line.replace("Model:","").replace("← BEST","⭐ BEST").strip()}
            for metric in ["Accuracy","Precision","Recall","F1 Score","AUC-ROC","CV AUC"]:
                key = metric.replace(" ","").lower()[:6]
                if f"  {metric}" in line:
                    val = line.split(":")[-1].strip().split("±")[0].strip()
                    current[metric] = val
        if current: rows.append(current)
        if rows:
            tbl = pd.DataFrame(rows).set_index("Model")
            st.dataframe(tbl.style.highlight_max(axis=0, color="#06D6A033"),
                         use_container_width=True)
    else:
        st.info("Run `python setup.py` first to train models and generate the report.")

# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ═══════════════════════════════════════════════════════════════════
def page_predict(df):
    st.title("🔮 Predict Job Acceptance")
    st.caption("Enter candidate details to predict job acceptance probability")

    model_path = MODEL_DIR / "best_model.pkl"
    feat_path  = MODEL_DIR / "feature_names.pkl"

    if not model_path.exists():
        st.warning("⚠️ No trained model found. Run `python setup.py` first.")
        return

    model      = joblib.load(model_path)
    feat_names = joblib.load(feat_path)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"<div style='color:{C['accent']};font-weight:700'>🎓 Academic</div>", unsafe_allow_html=True)
        gpa   = st.slider("GPA (4–10)", 4.0, 10.0, 7.5, 0.1)
        tenth = st.slider("10th %",     40.0, 100.0, 75.0, 0.5)
        twlft = st.slider("12th %",     40.0, 100.0, 73.0, 0.5)
        bklgs = st.number_input("Backlogs", 0, 5, 0)

    with c2:
        st.markdown(f"<div style='color:{C['info']};font-weight:700'>💼 Experience & Skills</div>", unsafe_allow_html=True)
        yrs_exp   = st.slider("Years Experience", 0.0, 15.0, 1.5, 0.5)
        num_sk    = st.slider("Number of Skills", 1, 10, 4)
        skill_pct = st.slider("Skills Match %",   5.0, 100.0, 65.0, 1.0)
        has_cert  = st.selectbox("Certification", ["None","AWS","Google Cloud","Azure","Coursera ML","PMP"])
        projects  = st.number_input("Projects Done", 0, 10, 2)

    with c3:
        st.markdown(f"<div style='color:{C['primary']};font-weight:700'>🎤 Interview & Job</div>", unsafe_allow_html=True)
        int_sc    = st.slider("Interview Score",     20.0, 100.0, 65.0, 0.5)
        apt_sc    = st.slider("Aptitude Score",      10.0, 100.0, 60.0, 0.5)
        empl_sc   = st.slider("Employability Score", 10.0, 100.0, 60.0, 0.5)
        comp_tier = st.selectbox("Company Tier",     ["Tier 1","Tier 2","Tier 3"])
        comp_lvl  = st.selectbox("Competition",      ["Low","Medium","High"])
        loc_match = st.selectbox("Location Match",   ["Yes","No"])

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Acceptance", use_container_width=True, type="primary")

    if predict_btn:
        from features.engineering import add_features
        from sklearn.preprocessing import LabelEncoder

        gpa_sc    = gpa * 10
        acad_sc   = tenth*0.25 + twlft*0.25 + gpa_sc*0.50
        cert_flag = 0 if has_cert == "None" else 1
        loc_num   = 1 if loc_match == "Yes" else 0
        salary    = {"Tier 1":1800000,"Tier 2":1000000,"Tier 3":550000}[comp_tier]
        bl        = int(bklgs)
        pp        = float(np.clip(
            int_sc*0.30 + skill_pct*0.25 + acad_sc*0.20
            + empl_sc*0.15 + cert_flag*10*0.10 - bl*3, 0, 100))
        high_risk = int(pp < 45 or int_sc < 40 or bl >= 2)

        # Encode categoricals same as training
        tier_enc  = {"Tier 1":0,"Tier 2":1,"Tier 3":2}.get(comp_tier, 1)
        comp_enc  = {"High":0,"Low":1,"Medium":2}.get(comp_lvl, 1)
        exp_cat   = "Fresher" if yrs_exp==0 else ("Junior" if yrs_exp<=2 else ("Mid-Level" if yrs_exp<=5 else "Senior"))
        exp_enc   = {"Fresher":0,"Junior":1,"Mid-Level":2,"Senior":3}.get(exp_cat, 0)
        acad_band = "Poor" if acad_sc<55 else ("Below Avg" if acad_sc<65 else ("Average" if acad_sc<75 else ("Good" if acad_sc<85 else "Excellent")))
        band_enc  = {"Poor":0,"Below Avg":1,"Average":2,"Good":3,"Excellent":4}.get(acad_band,2)
        sk_lvl    = "Low" if skill_pct<40 else ("Medium" if skill_pct<60 else ("High" if skill_pct<80 else "Excellent"))
        sk_enc    = {"Low":0,"Medium":1,"High":2,"Excellent":3}.get(sk_lvl, 1)

        row = {
            "tenth_percentage":tenth,"twelfth_percentage":twlft,"gpa":gpa,
            "backlogs":bl,"years_experience":yrs_exp,"num_skills":num_sk,
            "skills_match_pct":skill_pct,"projects_done":projects,"num_interviews":2,
            "interview_score":int_sc,"aptitude_score":apt_sc,"employability_score":empl_sc,
            "offered_salary_inr":salary,"location_pref_match":loc_num,"age":24,
            "has_certification":cert_flag,"academic_score":acad_sc,
            "placement_probability":pp,"high_risk":high_risk,
            "gender_enc":0,"degree_enc":0,"certification_enc":0,
            "company_tier_enc":tier_enc,"job_role_enc":0,"competition_level_enc":comp_enc,
            "experience_category_enc":exp_enc,"academic_band_enc":band_enc,
            "skills_level_enc":sk_enc,
        }
        X_pred = pd.DataFrame([row])
        X_pred = X_pred.reindex(columns=feat_names, fill_value=0)

        prob  = model.predict_proba(X_pred)[0][1]
        pred  = "✅ PLACED" if prob >= 0.5 else "❌ NOT PLACED"
        color = C["success"] if prob >= 0.5 else C["danger"]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.markdown(f"""<div class="kpi-card">
          <div class="kpi-val" style="color:{color}">{pred}</div>
          <div class="kpi-lbl">Prediction</div></div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class="kpi-card">
          <div class="kpi-val" style="color:{color}">{prob*100:.1f}%</div>
          <div class="kpi-lbl">Acceptance Probability</div></div>""", unsafe_allow_html=True)
        risk_clr = C["danger"] if high_risk else C["success"]
        r3.markdown(f"""<div class="kpi-card">
          <div class="kpi-val" style="color:{risk_clr}">{"⚠️ HIGH" if high_risk else "✅ LOW"}</div>
          <div class="kpi-lbl">Risk Level</div></div>""", unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(prob*100,1),
            domain={"x":[0,1],"y":[0,1]},
            title={"text":"Acceptance Probability","font":{"color":C["primary"],"size":16}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":C["muted"]},
                "bar":{"color":color},
                "steps":[
                    {"range":[0,40],"color":C["danger"]+"44"},
                    {"range":[40,60],"color":C["accent"]+"44"},
                    {"range":[60,100],"color":C["success"]+"44"},
                ],
                "threshold":{"line":{"color":"white","width":3},"thickness":0.75,"value":50},
                "bgcolor":C["card"],"bordercolor":C["border"],
            },
            number={"suffix":"%","font":{"color":color,"size":28}},
        ))
        fig.update_layout(paper_bgcolor=C["bg"], height=300,
                          margin=dict(t=60,b=10,l=40,r=40))
        st.plotly_chart(fig, use_container_width=True)

        # Key factors
        st.markdown(f"<div class='section-hdr'>📋 Key Factors</div>", unsafe_allow_html=True)
        factors = [
            ("Interview Score",int_sc,50,100),
            ("Skills Match %",skill_pct,30,100),
            ("Academic Score",round(acad_sc,1),50,100),
            ("Employability Score",empl_sc,30,100),
            ("Placement Probability",pp,0,100),
        ]
        for fname2, val2, lo, hi in factors:
            norm = (val2-lo)/(hi-lo)*100
            clr2 = C["success"] if norm>60 else (C["accent"] if norm>40 else C["danger"])
            st.markdown(f"""
            <div style="margin:6px 0">
              <span style="color:{C['muted']};font-size:.85rem">{fname2}</span>
              <span style="color:{clr2};font-weight:700;float:right">{val2}</span>
              <div style="background:{C['border']};border-radius:4px;height:6px;margin-top:4px">
                <div style="background:{clr2};width:{norm:.0f}%;height:6px;border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    df, page = sidebar(df_all)

    if   "Overview" in page: page_overview(df)
    elif "EDA"      in page: page_eda(df)
    elif "Model"    in page: page_models()
    elif "Predict"  in page: page_predict(df)

    st.markdown(f"""<div style='text-align:center;color:{C['muted']};
        font-size:.72rem;padding:10px 0;border-top:1px solid {C['border']};margin-top:20px'>
        🎯 <strong style='color:{C['primary']}'>Job Acceptance Prediction System</strong>
        · GUVI HCL Fullstack Data Science Program · Python · Streamlit · Scikit-learn
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
