import os
from flask import Blueprint, render_template, request, redirect, url_for, current_app
from website.db_extension import db
from website.models import Document

documents_bp = Blueprint("documents", __name__)


@documents_bp.route("/documents", methods=["GET"])
def list_documents():
    documents = Document.query.order_by(Document.uploaded_at.desc()).all()
    return render_template("documents.html", documents=documents)


@documents_bp.route("/documents/upload", methods=["POST"])
def upload_document():
    file = request.files.get("file")

    if not file or file.filename == "":
        return redirect(url_for("documents.list_documents"))

    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    doc = Document(filename=file.filename, status="pending")
    db.session.add(doc)
    db.session.commit()

    return redirect(url_for("documents.list_documents"))