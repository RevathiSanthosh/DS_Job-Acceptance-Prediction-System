"""
setup.py — One-click full pipeline runner
Runs: Data Gen → Clean → Feature Eng → EDA (15 charts) → Train 6 Models → Best Model
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def banner(msg):
    print(f"\n{'═'*62}\n  {msg}\n{'═'*62}")

def main():
    t0 = time.time()
    print("""
╔════════════════════════════════════════════════════════════╗
║   🎯  JOB ACCEPTANCE PREDICTION SYSTEM                    ║
║   GUVI HCL · Fullstack Data Science Program               ║
╚════════════════════════════════════════════════════════════╝""")

    # STEP 1 ─ Data Generation
    banner("STEP 1/4 — Generating Synthetic Dataset (50,000 records)")
    from data.generator import generate
    raw = generate(save=True)
    print(f"  ✅ Raw dataset: {len(raw):,} rows × {raw.shape[1]} columns")

    # STEP 2 ─ Cleaning & Preprocessing
    banner("STEP 2/4 — Data Cleaning & Preprocessing (6 steps)")
    from preprocessing.pipeline import run_pipeline
    clean, rpt, scaler, encoders = run_pipeline(raw, verbose=True)
    print(f"  ✅ Clean shape: {clean.shape[0]:,} rows × {clean.shape[1]} columns")

    # STEP 3 ─ Feature Engineering
    banner("STEP 3/4 — Feature Engineering")
    from features.engineering import add_features
    clean = add_features(clean)

    # STEP 3b ─ EDA Charts
    banner("STEP 3b — Generating 15 EDA Visualizations")
    from eda.visualizations import run_all
    paths = run_all(clean)
    print(f"  ✅ {len(paths)} EDA charts saved")

    # STEP 4 ─ Train 6 Models
    banner("STEP 4/4 — Training 5 ML Algorithms & Selecting Best")
    from models.training import train_all
    results = train_all(clean)
    best    = results["best"]

    elapsed = time.time() - t0
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  ✅ PIPELINE COMPLETE  ({elapsed:.0f}s)
║                                                            ║
║  📁 outputs/eda_charts/  → 20 charts (15 EDA + 5 model)  ║
║  📁 outputs/model_plots/ → trained model files (.pkl)     ║
║  📁 outputs/reports/     → cleaning + model reports       ║
║  💾 outputs/job_placement_cleaned.csv                     ║
║  🏆 Best Model: {best:<44}║
║  AUC-ROC: {results[best]['auc']:.4f}                                      ║
║                                                            ║
║  🚀 Launch Dashboard:                                      ║
║     streamlit run dashboard/app.py                         ║
╚════════════════════════════════════════════════════════════╝""")

if __name__ == "__main__":
    main()
