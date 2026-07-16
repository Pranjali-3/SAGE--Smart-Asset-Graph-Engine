import os
import joblib
import pandas as pd
import logging
from data_processor import NASAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self):

        model_folder = "models"

        self.rul_model = joblib.load(
            os.path.join(model_folder, "rul_model.pkl")
        )

        self.failure_model = joblib.load(
            os.path.join(model_folder, "failure_model.pkl")
        )

        logger.info("Prediction models loaded successfully.")
        self.feature_columns = joblib.load(
        os.path.join(model_folder, "feature_columns.pkl")
        )
    def preprocess(self, data):

        if isinstance(data, pd.DataFrame):
            return data[self.feature_columns]

        if isinstance(data, dict):
            return pd.DataFrame([data])[self.feature_columns]
    def predict_rul(self, data):

        X = self.preprocess(data)

        prediction = self.rul_model.predict(X)

        return float(prediction[0])
    def predict_failure(self, data):

        X = self.preprocess(data)

        prediction = self.failure_model.predict(X)

        return prediction[0]
    def predict(self, data):

        rul = self.predict_rul(data)

        failure = self.predict_failure(data)

        return {

            "Remaining Useful Life": round(rul, 2),

            "Failure Status": failure

        }
if __name__ == "__main__":

    predictor = Predictor()

    processor = NASAProcessor(
    r"data/nasa/archive/CMaps/train_FD001.txt"
    )

    processor.load_dataset()
    processor.calculate_rul()

    sample = processor.prepare_prediction_sample(
        engine_id=1,
        cycle=101
    )

    actual_rul = processor.df[
        (processor.df.engine_id == 1) &
        (processor.df.cycle == 101)
    ]["RUL"].iloc[0]

    print("Actual:", actual_rul)

    result = predictor.predict(sample)

    print(result)