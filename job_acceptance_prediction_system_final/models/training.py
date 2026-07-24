"""
models/training.py
------------------
Trains 6 ML classifiers and selects the best by AUC-ROC:
  1. Logistic Regression
  2. K-Nearest Neighbors
  3. Decision Tree
  4. Random Forest
  5. Gradient Boosting
  6. Support Vector Machine (SVM)

Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, CV Score
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, roc_curve,
                                      confusion_matrix, classification_report)

from config.settings import MODEL_DIR, EDA_DIR, RPT_DIR, C, SEED, TEST_SIZE, TARGET

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "tenth_percentage","twelfth_percentage","gpa","backlogs","years_experience",
    "num_skills","skills_match_pct","projects_done","num_interviews",
    "interview_score","aptitude_score","employability_score",
    "offered_salary_inr","location_pref_match","age",
    "has_certification","academic_score","placement_probability","high_risk",
    "gender_enc","degree_enc","certification_enc",
    "company_tier_enc","job_role_enc","competition_level_enc",
]

plt.rcParams.update({
    "figure.facecolor":C["bg"],"axes.facecolor":C["card"],
    "axes.edgecolor":C["border"],"text.color":C["text"],
    "axes.labelcolor":C["muted"],"xtick.color":C["muted"],
    "ytick.color":C["muted"],"grid.color":C["border"],
})


def _get_xy(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    # fill any remaining NaN with 0 (after scaling, remaining are structural)
    X = X.fillna(0)
    y = df[TARGET+"_num"].astype(int)
    return X, y, available


def _define_models():
    return {
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.8,
                                       random_state=SEED, class_weight="balanced")),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("sc", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10,
            random_state=SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=SEED),
    }


def train_all(df: pd.DataFrame) -> dict:
    X, y, feat_names = _get_xy(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    print(f"\n  🎯  Training set : {len(X_tr):,}   Test set : {len(X_te):,}")
    print(f"      Features     : {X_tr.shape[1]}")
    placed = int(y.sum()); total = len(y)
    print(f"      Class dist   : Placed={placed:,} ({placed/total*100:.1f}%)  "
          f"Not Placed={total-placed:,} ({(total-placed)/total*100:.1f}%)\n")

    models  = _define_models()
    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        print(f"  ⏳  {name}...")
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_prob  = model.predict_proba(X_te)[:, 1]
        cv_sc   = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

        acc = accuracy_score(y_te, y_pred)
        pre = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1  = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_prob)
        fpr, tpr, _ = roc_curve(y_te, y_prob)

        results[name] = {
            "model"      : model,
            "accuracy"   : round(acc, 4),
            "precision"  : round(pre, 4),
            "recall"     : round(rec, 4),
            "f1"         : round(f1,  4),
            "auc"        : round(auc, 4),
            "cv_auc_mean": round(cv_sc.mean(), 4),
            "cv_auc_std" : round(cv_sc.std(),  4),
            "fpr"        : fpr,
            "tpr"        : tpr,
            "confusion"  : confusion_matrix(y_te, y_pred),
            "report"     : classification_report(y_te, y_pred,
                             target_names=["Not Placed","Placed"]),
            "feat_names" : feat_names,
        }
        print(f"      Acc={acc:.4f}  Pre={pre:.4f}  Rec={rec:.4f}  "
              f"F1={f1:.4f}  AUC={auc:.4f}  CV-AUC={cv_sc.mean():.4f}±{cv_sc.std():.4f}")

        joblib.dump(model, MODEL_DIR / f"{name.replace(' ','_').lower()}.pkl")

    # ── Select best model by AUC ──────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  🏆  BEST MODEL: {best_name}  (AUC={results[best_name]['auc']:.4f})")

    # ── Feature importance for tree-based / coef for LR ──────────
    best_model = results[best_name]["model"]
    clf = getattr(best_model, "named_steps", {}).get("clf", best_model)
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
    elif hasattr(clf, "coef_"):
        imp = pd.Series(np.abs(clf.coef_[0]), index=feat_names).sort_values(ascending=False)
    else:
        imp = pd.Series(np.ones(len(feat_names)), index=feat_names)
    results["feature_importance"] = imp
    results["best"]               = best_name
    results["feature_names"]      = feat_names

    joblib.dump(results[best_name]["model"], MODEL_DIR / "best_model.pkl")
    joblib.dump(feat_names,                 MODEL_DIR / "feature_names.pkl")

    _plot_all(results)
    _save_report(results)
    return results


# ── Evaluation plots ──────────────────────────────────────────────
def _plot_all(results):
    model_names = [k for k in results
                   if k not in ("best","feature_importance","feature_names")]
    colors = [C["primary"],C["success"],C["accent"],
              C["danger"],C["info"],"#AA44FF"]

    # ── A) ROC Curves ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C["bg"])
    for (nm, clr) in zip(model_names, colors):
        r = results[nm]
        ax.plot(r["fpr"], r["tpr"], color=clr, lw=2.5,
                label=f"{nm}  (AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1],"--",color=C["muted"],lw=1)
    ax.set_xlabel("False Positive Rate",color=C["muted"])
    ax.set_ylabel("True Positive Rate",color=C["muted"])
    ax.set_title("ROC Curves — All 6 Models",color=C["primary"],fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.4)
    ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(EDA_DIR/"16_roc_curves.png",dpi=150,bbox_inches="tight",facecolor=C["bg"])
    plt.close(fig); print("  💾 16_roc_curves.png")

    # ── B) Model Comparison ───────────────────────────────────────
    metrics = ["accuracy","precision","recall","f1","auc","cv_auc_mean"]
    metric_labels = ["Accuracy","Precision","Recall","F1","AUC-ROC","CV AUC"]
    x = np.arange(len(metrics)); w = 0.13
    fig, ax = plt.subplots(figsize=(15, 6), facecolor=C["bg"])
    for i, (nm, clr) in enumerate(zip(model_names, colors)):
        vals = [results[nm][m] for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=nm, color=clr, alpha=0.85, edgecolor=C["bg"])
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.004, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color=C["text"])
    ax.set_xticks(x + w*2.5); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score", color=C["muted"])
    ax.set_title("Model Performance Comparison — All 6 Algorithms",
                 color=C["primary"], fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.4)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.2)
    fig.tight_layout()
    fig.savefig(EDA_DIR/"17_model_comparison.png",dpi=150,bbox_inches="tight",facecolor=C["bg"])
    plt.close(fig); print("  💾 17_model_comparison.png")

    # ── C) Feature Importance ─────────────────────────────────────
    if "feature_importance" in results:
        imp = results["feature_importance"].head(15)
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=C["bg"])
        clrs = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp)))
        ax.barh(imp.index[::-1], imp.values[::-1], color=clrs, edgecolor=C["bg"])
        ax.set_xlabel("Feature Importance",color=C["muted"])
        ax.set_title(f"Top 15 Features — {results['best']}",
                     color=C["primary"],fontweight="bold")
        ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",alpha=0.2)
        fig.tight_layout()
        fig.savefig(EDA_DIR/"18_feature_importance.png",dpi=150,bbox_inches="tight",facecolor=C["bg"])
        plt.close(fig); print("  💾 18_feature_importance.png")

    # ── D) Confusion Matrix (best) ────────────────────────────────
    best = results["best"]
    cm   = results[best]["confusion"]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=C["bg"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Placed","Placed"],
                yticklabels=["Not Placed","Placed"],
                linewidths=1, linecolor=C["bg"], ax=ax,
                annot_kws={"size":14,"weight":"bold"})
    ax.set_xlabel("Predicted",color=C["muted"])
    ax.set_ylabel("Actual",color=C["muted"])
    ax.set_title(f"Confusion Matrix — {best}",color=C["primary"],fontweight="bold")
    fig.tight_layout()
    fig.savefig(EDA_DIR/"19_confusion_matrix.png",dpi=150,bbox_inches="tight",facecolor=C["bg"])
    plt.close(fig); print("  💾 19_confusion_matrix.png")

    # ── E) AUC Ranking bar ───────────────────────────────────────
    auc_vals = {nm: results[nm]["auc"] for nm in model_names}
    auc_s = dict(sorted(auc_vals.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C["bg"])
    clrs2 = [C["success"] if nm==results["best"] else C["primary"]
             for nm in auc_s]
    bars = ax.barh(list(auc_s.keys())[::-1], list(auc_s.values())[::-1],
                   color=clrs2[::-1], edgecolor=C["bg"], height=0.55)
    for b in bars:
        w2 = b.get_width()
        ax.text(w2+0.002, b.get_y()+b.get_height()/2,
                f"{w2:.4f}", va="center", fontsize=10, color=C["text"])
    ax.set_xlim(0, 1.12); ax.set_xlabel("AUC-ROC",color=C["muted"])
    ax.set_title("Algorithm Ranking by AUC-ROC  (🟢 = Best)",
                 color=C["primary"],fontweight="bold")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",alpha=0.2)
    ax.axvline(0.5, color=C["danger"], lw=1, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(EDA_DIR/"20_algorithm_ranking.png",dpi=150,bbox_inches="tight",facecolor=C["bg"])
    plt.close(fig); print("  💾 20_algorithm_ranking.png")


def _save_report(results):
    model_names = [k for k in results
                   if k not in ("best","feature_importance","feature_names")]
    lines = ["JOB ACCEPTANCE — MODEL EVALUATION REPORT","="*60,
             f"\nBEST MODEL : {results['best']}",""]
    for nm in model_names:
        r = results[nm]
        best_flag = " ← BEST" if nm == results["best"] else ""
        lines += [f"\n{'─'*55}",f"Model: {nm}{best_flag}",
                  f"  Accuracy    : {r['accuracy']:.4f}",
                  f"  Precision   : {r['precision']:.4f}",
                  f"  Recall      : {r['recall']:.4f}",
                  f"  F1 Score    : {r['f1']:.4f}",
                  f"  AUC-ROC     : {r['auc']:.4f}",
                  f"  CV AUC (5K) : {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f}",
                  f"\nClassification Report:\n{r['report']}"]
    (RPT_DIR/"model_evaluation.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄 Model report → {RPT_DIR/'model_evaluation.txt'}")


if __name__ == "__main__":
    from data.generator import generate
    from preprocessing.pipeline import run_pipeline
    from features.engineering import add_features
    raw = generate(save=False)
    clean, *_ = run_pipeline(raw, verbose=False)
    clean = add_features(clean)
    results = train_all(clean)
    print(f"\nBest: {results['best']}")
