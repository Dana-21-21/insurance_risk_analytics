# src/config.py

from pathlib import Path

# ============================
# Project Paths
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "MachineLearningRating_v3.txt"

REPORTS_PATH = PROJECT_ROOT / "reports"

MODELS_PATH = PROJECT_ROOT / "models"

# ============================
# Visualization
# ============================

FIGSIZE = (10, 6)

RANDOM_STATE = 42

# ============================
# Columns
# ============================

DATE_COLUMN = "TransactionMonth"

TARGET_PREMIUM = "TotalPremium"

TARGET_CLAIMS = "TotalClaims"