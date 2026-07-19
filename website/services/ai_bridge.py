from ai_core.retriever import Retriever
from ai_core.predict import Predictor
from ai_core.explain_prediction import PredictionExplainer
from ai_core.recommendation import RecommendationEngine
from ai_core.knowledge_graph import KnowledgeGraph
from ai_core.data_processor import NASAProcessor

_predictor = None
_explainer = None
_recommender = None
_kg = None
_retriever = None


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


def run_prediction_pipeline(engine_id: int, cycle: int, dataset_path: str):
    try:
        processor = NASAProcessor(dataset_path)
        processor.load_dataset()
        processor.clean_dataset()
        processor.normalize_dataset()
        processor.calculate_health_index()
        processor.calculate_rul()
        processor.generate_failure_labels()
        processor.calculate_rolling_features()

        row = processor.df[
            (processor.df.engine_id == engine_id) &
            (processor.df.cycle == cycle)
        ]
        if row.empty:
            return {"success": False, "error": "No matching engine_id/cycle found."}

        sample = row.iloc[0].to_dict()

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


def entity_report(entity_name: str):
    try:
        kg = get_kg()
        if not kg.entity_exists(entity_name):
            return {"success": False, "error": "Entity not found"}
        return {"success": True, "report": kg.failure_report(entity_name)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def graph_stats():
    try:
        return {"success": True, "stats": get_kg().graph_statistics()}
    except Exception as e:
        return {"success": False, "error": str(e)}