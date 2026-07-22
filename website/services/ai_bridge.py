from ai_core.retriever import Retriever
from ai_core.predict import Predictor
from ai_core.explain_prediction import PredictionExplainer
from ai_core.recommendation import RecommendationEngine
from ai_core.knowledge_graph import KnowledgeGraph
from ai_core.data_processor import NASAProcessor
from ai_core.copilot import Copilot

_predictor = None
_explainer = None
_recommender = None
_kg = None
_retriever = None
_copilot = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = PredictionExplainer()
    return _explainer


def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = RecommendationEngine()
    return _recommender


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

def reload_retriever():
    global _retriever
    if _retriever is not None:
        _retriever.reload()


def get_kg():
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
        _kg.import_graphml("knowledge_graph.graphml")
    return _kg


def search_knowledge(query: str, top_k=10):
    try:
        results = get_retriever().retrieve(query, top_k=top_k)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_prediction_pipeline(machine_id: int, dataset_path: str, equipment_type: str = "engine"):
    try:
        # For engines, try to get real data from dataset
        if equipment_type == "engine":
            processor = NASAProcessor(dataset_path)
            processor.load_dataset()
            processor.clean_dataset()
            processor.normalize_dataset()
            processor.calculate_health_index()
            processor.calculate_rul()
            processor.generate_failure_labels()
            processor.calculate_rolling_features()

            engine_data = processor.df[
                processor.df.engine_id == machine_id
            ]

            if not engine_data.empty:
                mid = len(engine_data) // 2
                row = engine_data.iloc[[mid]]
                sample = row.iloc[0].to_dict()
            else:
                # Engine not in dataset, use default sample
                sample = _create_default_sample(machine_id)
        else:
            # For sensors, pumps, valves, etc. - use default sample
            sample = _create_default_sample(machine_id)

        prediction = get_predictor().predict(sample)
        reasons = get_explainer().top_reasons()
        recommendation = get_recommender().recommend(prediction, reasons)

        return {
            "success": True,
            "prediction": prediction,
            "top_reasons": reasons,
            "recommendation": recommendation,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _create_default_sample(machine_id: int):
    """Create a realistic sensor sample based on training data statistics, normalized like training."""
    import random
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    random.seed(machine_id)

    # Load actual training data to get realistic sensor ranges
    try:
        df = pd.read_csv(
            os.path.join("data", "nasa", "archive", "CMaps", "train_FD001.txt"),
            sep=r"\s+",
            header=None,
            names=["engine_id", "cycle", "setting1", "setting2", "setting3"] +
                  [f"sensor_{i}" for i in range(1, 22)]
        )
        # Normalize sensors like training pipeline
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        scaler = MinMaxScaler()
        df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

        # Get healthy engines (early cycles) after normalization
        healthy = df[df["cycle"] <= 20]
    except:
        healthy = None
        scaler = None

    sample = {}

    # Settings - use small values like normalized training data
    for setting in ["setting1", "setting2", "setting3"]:
        if healthy is not None and setting in healthy.columns:
            sample[setting] = float(healthy[setting].mean())
        else:
            sample[setting] = 0.0

    # Sensors - use NORMALIZED values from training data
    for i in range(1, 22):
        col = f"sensor_{i}"
        if healthy is not None and col in healthy.columns:
            mean_val = float(healthy[col].mean())
            std_val = float(healthy[col].std())
            # Add small random variation around healthy mean
            val = mean_val + random.gauss(0, std_val * 0.1)
            # Clamp to [0, 1] range (normalized)
            sample[col] = max(0.0, min(1.0, val))
        else:
            sample[col] = 0.5

    # Add rolling features (mean and std for each sensor)
    for i in range(1, 22):
        col = f"sensor_{i}"
        sample[f"{col}_mean"] = sample[col]
        sample[f"{col}_std"] = 0.01

    # Add health index
    sensor_vals = [sample.get(f"sensor_{i}", 0.5) for i in range(1, 22)]
    sample["health_index"] = 1 - (sum(sensor_vals) / len(sensor_vals))

    return sample


def entity_report(entity_name: str):
    try:
        kg = get_kg()
        entity = kg.find_entity(entity_name)

        if entity is None:
            return {
                "success": False,
                "error": "Entity not found"
            }

        return {
            "success": True,
            "report": kg.failure_report(entity)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def graph_stats():
    try:
        return {"success": True, "stats": get_kg().graph_statistics()}
    except Exception as e:
        return {"success": False, "error": str(e)}
    
def get_copilot():
    global _copilot
    if _copilot is None:
        _copilot = Copilot()
    return _copilot

def ask_copilot(question: str):
    try:
        result = get_copilot().ask(question)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}  