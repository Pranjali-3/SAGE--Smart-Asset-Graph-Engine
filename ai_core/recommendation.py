import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    SENSOR_ACTIONS = {

    "Fuel Flow": "Inspect fuel injectors and fuel delivery system.",

    "Vibration": "Check bearings, shafts and rotor alignment.",

    "Engine Efficiency": "Run engine performance diagnostics.",

    "Oil Temperature": "Inspect lubrication and cooling system.",

    "Compressor Temperature": "Inspect compressor and airflow.",

    "Core Temperature": "Check cooling fan and heat dissipation.",

    "Fuel Pressure": "Inspect fuel pump and pressure regulator.",

    "Bearing Temperature": "Check bearing lubrication and wear."
}

    def __init__(self):

        logger.info("Recommendation Engine initialized.")
    def rul_recommendation(self, rul):

        if rul <= 20:

            return {
                "priority": "High",
                "recommendation": "Immediate maintenance required."
            }

        elif rul <= 50:

            return {
                "priority": "Medium",
                "recommendation": "Schedule maintenance soon."
            }

        else:

            return {
                "priority": "Low",
                "recommendation": "Engine is operating normally."
            }
    def failure_recommendation(self, status):

        if status == "Critical":

            return [
                "Inspect engine immediately.",
                "Replace worn components.",
                "Perform complete diagnostics."
            ]

        elif status == "Warning":

            return [
                "Inspect abnormal sensors.",
                "Monitor engine frequently.",
                "Plan preventive maintenance."
            ]

        return [
            "Continue normal operation.",
            "Perform routine inspection."
        ]
    def recommend(self, prediction, reasons):

        rul = prediction["Remaining Useful Life"]
        status = prediction["Failure Status"]

        result = self.rul_recommendation(rul)

        actions = self.failure_recommendation(status)
        sensor_actions = self.explain_based_actions(reasons)
        all_actions = list(dict.fromkeys(actions + sensor_actions))

        return {

            "priority": result["priority"],

            "summary": result["recommendation"],

            "actions": all_actions
        }
    def explain_based_actions(self, reasons):

        actions = []

        for reason in reasons:

            feature = reason["feature"]

            if feature in self.SENSOR_ACTIONS:

                actions.append(self.SENSOR_ACTIONS[feature])

        return actions
if __name__ == "__main__":

    engine = RecommendationEngine()
    prediction = {
        "Remaining Useful Life": 28,
        "Failure Status": "Critical"
    }

    reasons = [
        {"feature": "Fuel Flow"},
        {"feature": "Vibration"},
        {"feature": "Oil Temperature"},
        {"feature": "Engine Efficiency"}
    ]

    recommendation = engine.recommend(
        prediction,
        reasons
    )

    print(recommendation)