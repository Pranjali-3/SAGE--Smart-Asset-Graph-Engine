import os
from flask import Blueprint, render_template, request, current_app
from website.services.ai_bridge import run_prediction_pipeline

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["GET", "POST"])
def predict():
    result = None

    if request.method == "POST":
        equipment_type = request.form.get("equipment_type", "engine")
        machine_id = int(request.form["machine_id"])

        dataset_path = os.path.join(
            "data", "nasa", "archive", "CMaps", "train_FD001.txt"
        )

        result = run_prediction_pipeline(
            machine_id,
            dataset_path,
            equipment_type
        )

        if result.get("success"):
            result["equipment_type"] = equipment_type
            result["machine_id"] = machine_id

    return render_template("predict.html", result=result)
