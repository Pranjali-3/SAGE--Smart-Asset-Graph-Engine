import os
import joblib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionExplainer:

    def __init__(self):

        model_folder = "models"

        self.feature_columns = joblib.load(
            os.path.join(model_folder, "feature_columns.pkl")
        )

        self.feature_importance = joblib.load(
            os.path.join(model_folder, "feature_importance.pkl")
        )

        logger.info("Prediction Explainer loaded.")

    SENSOR_NAMES = {

    "setting1": "Operational Setting 1",
    "setting2": "Operational Setting 2",
    "setting3": "Operational Setting 3",

    "sensor_1": "Fan Speed",
    "sensor_2": "Core Temperature",
    "sensor_3": "Compressor Pressure",
    "sensor_4": "Fuel Flow",
    "sensor_5": "Engine Temperature",
    "sensor_6": "Air Pressure",
    "sensor_7": "Rotor Speed",
    "sensor_8": "Exhaust Temperature",
    "sensor_9": "Vibration",
    "sensor_10": "Cooling Pressure",
    "sensor_11": "Oil Temperature",
    "sensor_12": "Oil Pressure",
    "sensor_13": "Fuel Pressure",
    "sensor_14": "Compressor Temperature",
    "sensor_15": "Turbine Temperature",
    "sensor_16": "Bearing Temperature",
    "sensor_17": "Rotor Vibration",
    "sensor_18": "Exhaust Pressure",
    "sensor_19": "Fuel Valve Position",
    "sensor_20": "Air Intake Flow",
    "sensor_21": "Engine Efficiency",

    "health_index": "Health Index"
    }
    def explain(self):

        importance = pd.DataFrame({

            "feature": self.feature_columns,

            "importance": self.feature_importance

        })

        importance = importance.sort_values(

            by="importance",

            ascending=False

        )

        return importance
    def top_reasons(self, top_n=5):

        importance = self.explain()

        reasons = []

        for _, row in importance.head(top_n).iterrows():

            feature = row["feature"]

            score = row["importance"]

            base_feature = feature

            if "_mean" in feature:
                base_feature = feature.replace("_mean", "")

            elif "_std" in feature:
                base_feature = feature.replace("_std", "")

            readable = self.SENSOR_NAMES.get(

                base_feature,

                base_feature

            )

            reasons.append({

                "feature": readable,

                "importance": round(score, 4)

            })

        return reasons
if __name__ == "__main__":

    explainer = PredictionExplainer()

    reasons = explainer.top_reasons()

    print()

    print("Top Factors")

    for reason in reasons:

        print(reason)