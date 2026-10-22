import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml.config import (
    PROCESSED_DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


def train_model():

    # --------------------------------------------------
    # 1. Load processed dataset
    # --------------------------------------------------

    print("Loading processed dataset...")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    print(f"Dataset shape: {df.shape}")


    # --------------------------------------------------
    # 2. Separate features (X) and target (y)
    # --------------------------------------------------

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]


    print("\nFeatures being used:")

    for feature in FEATURE_COLUMNS:
        print(f" - {feature}")

    print(f"\nTarget: {TARGET_COLUMN}")


    # --------------------------------------------------
    # 3. Define preprocessing pipelines
    # --------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )


    # --------------------------------------------------
    # 4. Combine numerical and categorical preprocessing
    # --------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


    # --------------------------------------------------
    # 5. Create Random Forest model
    # --------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


    # --------------------------------------------------
    # 6. Create complete ML pipeline
    # --------------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


    # --------------------------------------------------
    # 7. Split data into training and testing sets
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )


    print("\nTraining model...")


    # --------------------------------------------------
    # 8. Train the model
    # --------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------
    # 9. Make predictions on test data
    # --------------------------------------------------

    predictions = pipeline.predict(X_test)


    # --------------------------------------------------
    # 10. Evaluate model
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )


    print("\nMODEL RESULTS")
    print("----------------------")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


    # --------------------------------------------------
    # 11. Create model directory if necessary
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------
    # 12. Save trained pipeline
    # --------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )


    # --------------------------------------------------
    # 13. Save feature list
    # --------------------------------------------------

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURE_COLUMNS_PATH,
    )


    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )

    print(
        f"\nFeature columns saved to:\n"
        f"{FEATURE_COLUMNS_PATH}"
    )


# --------------------------------------------------
# Run training when this file is executed directly
# --------------------------------------------------

if __name__ == "__main__":
    train_model()