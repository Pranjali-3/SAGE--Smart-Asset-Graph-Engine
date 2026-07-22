import os
import joblib
import logging

import pandas as pd

from .dataset_manager import DatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Loading Prediction Models...")
        logger.info("=" * 60)

        model_folder = "models"

        self.rul_model = joblib.load(
            os.path.join(model_folder, "rul_model.pkl")
        )

        self.failure_model = joblib.load(
            os.path.join(model_folder, "failure_model.pkl")
        )

        self.feature_columns = joblib.load(
            os.path.join(model_folder, "feature_columns.pkl")
        )

        logger.info("Prediction models loaded.")

    # ======================================================
    # Feature Preparation
    # ======================================================

    def preprocess(self, data):

        if isinstance(data, dict):

            data = pd.DataFrame([data])

        if set(self.feature_columns).issubset(data.columns):

            return data[self.feature_columns]

        sensor_cols = [
            f"sensor_{i}"
            for i in range(1, 22)
        ]

        data["health_index"] = (
            1 -
            data[sensor_cols].mean(axis=1)
        )

        for i in range(1, 22):

            data[f"sensor_{i}_mean"] = data[f"sensor_{i}"]

            data[f"sensor_{i}_std"] = 0.0

        return data[self.feature_columns]

    # ======================================================
    # Predict RUL
    # ======================================================

    def predict_rul(self, sample):

        X = self.preprocess(sample)

        return float(
            self.rul_model.predict(X)[0]
        )

    # ======================================================
    # Predict Failure
    # ======================================================

    def predict_failure(self, sample):

        X = self.preprocess(sample)

        return self.failure_model.predict(X)[0]

    # ======================================================
    # Unified Prediction
    # ======================================================

    def predict(self, sample):

        predicted_rul = self.predict_rul(sample)

        predicted_failure = self.predict_failure(sample)

        return {
            "Remaining Useful Life": round(predicted_rul, 2),
            "Failure Status": predicted_failure
        }

    # ======================================================
    # Predict Engine
    # ======================================================

    def predict_engine(self, engine_info):

        sample = engine_info["sample"]

        predicted_rul = self.predict_rul(sample)

        predicted_failure = self.predict_failure(sample)

        return {

            "Dataset": engine_info["dataset"],

            "Engine": engine_info["engine"],

            "Cycle": engine_info["cycle"],

            "Actual RUL": round(engine_info["actual_rul"], 2),

            "Predicted RUL": round(predicted_rul, 2),

            "Failure Status": predicted_failure

        }


# ======================================================
# Demo
# ======================================================

if __name__ == "__main__":

    manager = DatasetManager()

    predictor = Predictor()

    print("\nIndustrial Prediction Engine")
    print("=" * 60)

    while True:

        value = input(
            "\nEnter Engine ID (exit to quit): "
        )

        if value.lower() == "exit":

            break

        try:

            engine_id = int(value)

        except ValueError:

            print("Invalid engine id.")

            continue

        engine = manager.get_engine_sample(engine_id)

        if engine is None:

            print("Engine not found.")

            continue

        result = predictor.predict_engine(engine)

        print()

        print("=" * 60)
        print("Prediction")
        print("=" * 60)

        for key, value in result.items():

            print(f"{key:18}: {value}")

        print("=" * 60)