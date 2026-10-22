from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "NER_Landslide_Prototype_Dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "landslide_processed.csv"
MODEL_DIR = BASE_DIR / "data" / "models"


# --------------------------------------------------
# TARGET
# --------------------------------------------------

TARGET_COLUMN = "Risk_Score"


# --------------------------------------------------
# GEOGRAPHIC / REGIONAL FEATURES
# --------------------------------------------------

CATEGORICAL_FEATURES = [
    "State",
    "District",
]


# --------------------------------------------------
# NUMERICAL FEATURES
# --------------------------------------------------

NUMERICAL_FEATURES = [
    "Rainfall_24h_mm",
    "Rainfall_72h_mm",
    "Soil_Moisture_pct",
    "Soil_Depth_cm",
    "Elevation_m",
    "Slope_deg",
    "NDVI",
    "Geology_Risk_Score",
    "Groundwater_m",
    "Earthquake_Magnitude",
    "Previous_Landslide",
]


FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# --------------------------------------------------
# IDENTIFIER
# --------------------------------------------------

ID_COLUMNS = [
    "Sample_ID"
]


# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "landslide_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl" 