import os
from flask import Blueprint, render_template, request, current_app
from ..services.ai_bridge import run_prediction_pipeline

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["GET", "POST"])
def predict():
    result = None

    if request.method == "POST":
        engine_id = int(request.form["engine_id"])
        cycle = int(request.form["cycle"])

        dataset_path = os.path.join(
            "data", "nasa", "archive", "CMaps", "train_FD001.txt"
        )

        result = run_prediction_pipeline(engine_id, cycle, dataset_path)

    return render_template("predict.html", result=result)