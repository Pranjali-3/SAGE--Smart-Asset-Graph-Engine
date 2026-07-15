import os
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data_processor import NASAProcessor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class ModelTrainer:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.df = None

        self.rul_model = None

        self.failure_model = None

        logger.info("Model Trainer initialized.")
    def prepare_dataset(self):

        processor = NASAProcessor(self.dataset_path)

        processor.load_dataset()

        processor.clean_dataset()

        processor.normalize_dataset()

        processor.calculate_health_index()

        processor.calculate_rul()

        processor.generate_failure_labels()

        processor.calculate_rolling_features()

        processor.detect_sensor_trends()

        processor.detect_anomalies()

        self.df = processor.df

        logger.info("Dataset prepared successfully.")
    def get_features(self):

        ignore_columns = [
            "engine_id",
            "cycle",
            "RUL",
            "failure_label"
        ]

        feature_columns = [

            col

            for col in self.df.columns

            if (
                col not in ignore_columns
                and not col.endswith("_trend")
                and not col.endswith("_status")
            )
        ]

        X = self.df[feature_columns]

        y_rul = self.df["RUL"]

        y_failure = self.df["failure_label"]

        return X, y_rul, y_failure
    def split_dataset(self):

        X, y_rul, y_failure = self.get_features()

        X_train, X_test, y_rul_train, y_rul_test = train_test_split(
            X,
            y_rul,
            test_size=0.2,
            random_state=42
        )

        _, _, y_failure_train, y_failure_test = train_test_split(
            X,
            y_failure,
            test_size=0.2,
            random_state=42
        )

        return (
            X_train,
            X_test,
            y_rul_train,
            y_rul_test,
            y_failure_train,
            y_failure_test
        )
    def train_rul_model(self):

        logger.info("Training Random Forest Regressor...")

        (
            X_train,
            X_test,
            y_rul_train,
            y_rul_test,
            _,
            _
        ) = self.split_dataset()

        self.rul_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        print("\nFeature dtypes:")
        print(X_train.dtypes)

        print("\nNon-numeric columns:")
        print(X_train.select_dtypes(exclude=["number"]).columns.tolist())

        self.rul_model.fit(
            X_train,
            y_rul_train
        )

        predictions = self.rul_model.predict(X_test)

        mae = mean_absolute_error(
            y_rul_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_rul_test,
                predictions
            )
        )

        r2 = r2_score(
            y_rul_test,
            predictions
        )

        print("\n========== RUL MODEL ==========")
        print(f"MAE  : {mae:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"R²   : {r2:.4f}")

        logger.info("RUL model trained successfully.")

    def train_failure_model(self):

        logger.info("Training Random Forest Classifier...")

        (
            X_train,
            X_test,
            _,
            _,
            y_failure_train,
            y_failure_test
        ) = self.split_dataset()

        self.failure_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        self.failure_model.fit(
            X_train,
            y_failure_train
        )

        predictions = self.failure_model.predict(X_test)

        accuracy = accuracy_score(
            y_failure_test,
            predictions
        )

        print("\n========== FAILURE MODEL ==========")
        print(f"Accuracy : {accuracy:.4f}")

        print("\nClassification Report")
        print(
            classification_report(
                y_failure_test,
                predictions
            )
        )

        print("\nConfusion Matrix")
        print(
            confusion_matrix(
                y_failure_test,
                predictions
            )
        )

        logger.info("Failure model trained successfully.")

    def save_models(self):

        os.makedirs("models", exist_ok=True)

        joblib.dump(
            self.rul_model,
            "models/rul_model.pkl"
        )

        joblib.dump(
            self.failure_model,
            "models/failure_model.pkl"
        )

        logger.info("Models saved successfully.")
if __name__ == "__main__":

    trainer = ModelTrainer(
        r"data/nasa/archive/CMaps/train_FD001.txt"
    )

    trainer.prepare_dataset()

    trainer.train_rul_model()

    trainer.train_failure_model()

    trainer.save_models()