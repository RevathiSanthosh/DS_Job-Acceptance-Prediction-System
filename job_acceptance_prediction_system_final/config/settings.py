"""config/settings.py — Central configuration for Job Acceptance Prediction System"""
import sys
from pathlib import Path

# ── Windows console safety net ─────────────────────────────────────
# Windows terminals often default to legacy codepages (cp1252/cp437)
# that cannot encode emoji or special characters (₹, →, ±, etc.).
# Reconfiguring stdout/stderr to UTF-8 here protects every print()
# statement across the whole project, no matter which script is run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs"
EDA_DIR    = OUT_DIR / "eda_charts"
MODEL_DIR  = OUT_DIR / "model_plots"
RPT_DIR    = OUT_DIR / "reports"
DB_PATH    = ROOT / "job_placement.db"

for d in [DATA_DIR, OUT_DIR, EDA_DIR, MODEL_DIR, RPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED       = 42
N_ROWS     = 50_000
TARGET     = "status"
TEST_SIZE  = 0.20

# ── Brand colours (dark HR theme) ────────────────────────────────
C = {
    "bg"      : "#0D1117",
    "card"    : "#161B22",
    "border"  : "#21262D",
    "primary" : "#6C63FF",   # indigo
    "accent"  : "#F7B731",   # gold
    "success" : "#06D6A0",   # teal-green
    "danger"  : "#EF476F",   # rose
    "info"    : "#48CAE4",   # sky
    "text"    : "#E6EDF3",
    "muted"   : "#8B949E",
    "white"   : "#FFFFFF",
}

# Plotly template colours for consistent charts
PLOTLY_COLORS = [C["primary"], C["success"], C["accent"],
                 C["danger"], C["info"], "#AA44FF", "#FF6B35"]
