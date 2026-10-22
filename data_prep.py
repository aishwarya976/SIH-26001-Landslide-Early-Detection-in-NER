import pandas as pd

from ml.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMNS,
)

def prepare_data():

    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
        + ID_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    # --------------------------------------------------
    # Remove duplicate records
    # --------------------------------------------------

    df = df.drop_duplicates()

    # --------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------

    for column in ["State", "District"]:
        df[column] = df[column].astype(str).str.strip()

    # --------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------

    numerical_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in ["State", "District"]
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Target must be numeric
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"Processed dataset saved to: "
        f"{PROCESSED_DATA_PATH}"
    )

    print(
        f"Final dataset shape: {df.shape}"
    )
    return df

if __name__ == "__main__":
    df=prepare_data()

    print(df.head())
    print(df["State"].unique())
    print(df["District"].unique())