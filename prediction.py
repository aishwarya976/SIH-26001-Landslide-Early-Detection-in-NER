import joblib
import pandas as pd

from ml.config import (
    MODEL_PATH,
    FEATURE_COLUMNS,
)


# Store the loaded model in memory
_model = None


def load_model():
    global _model

    if _model is None:
        print("Loading landslide model...")
        _model = joblib.load(MODEL_PATH)

    return _model


def classify_risk(score: float) -> str:
    """
    Prototype risk classification.

    These thresholds are temporary.
    They must be calibrated using the actual
    Risk_Score definition and domain methodology
    before being used for real warnings.
    """

    if score < 0.40:
        return "LOW"

    elif score < 0.70:
        return "MODERATE"

    else:
        return "HIGH"


def predict_risk(data: dict):
    """
    Generate a landslide risk prediction
    from the supplied environmental and
    regional features.
    """

    # Load the trained model
    model = load_model()


    # Check that all features required by
    # the trained model are present
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Missing input features: {missing_features}"
        )


    # Keep only the features used by the model
    input_data = {
        feature: data[feature]
        for feature in FEATURE_COLUMNS
    }


    # Convert the input into a one-row DataFrame
    df = pd.DataFrame([input_data])


    # Generate prediction
    predicted_score = float(
        model.predict(df)[0]
    )


    # Convert the predicted score into
    # a human-readable risk category
    risk_level = classify_risk(
        predicted_score
    )


    # Return prediction
    return {
        "risk_score": round(
            predicted_score,
            4
        ),
        "risk_level": risk_level,
    }