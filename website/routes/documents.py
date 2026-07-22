import os

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    current_app
)

from website.db_extension import db
from website.models import Document
from website.services.ai_bridge import reload_retriever, get_kg
from ai_core.embeddings import index_document, remove_document
from ai_core.ingestion import ingest_document


documents_bp = Blueprint("documents", __name__)


@documents_bp.route("/documents", methods=["GET"])
def list_documents():

    documents = Document.query.order_by(
        Document.uploaded_at.desc()
    ).all()

    return render_template(
        "documents.html",
        documents=documents
    )


@documents_bp.route("/documents/upload", methods=["POST"])
def upload_document():

    file = request.files.get("file")

    if not file or file.filename == "":
        return redirect(url_for("documents.list_documents"))

    # -------------------------------------------------
    # Save uploaded file
    # -------------------------------------------------

    save_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(save_path)

    status = "indexed"

    try:

        # =============================================
        # 1. Add to FAISS
        # =============================================

        index_document(save_path)
        reload_retriever()

        # =============================================
        # 2. Read document
        # =============================================

        document = ingest_document(save_path)

        if isinstance(document, list):

            text = "\n\n".join(
                chunk["text"]
                for chunk in document
            )

        else:

            text = str(document)

        # =============================================
        # 3. Update Knowledge Graph
        # =============================================

        kg = get_kg()

        kg.add_document(text)

    except Exception as e:

        print("UPLOAD ERROR:", e)

        status = "failed"

    # -------------------------------------------------
    # Save document record
    # -------------------------------------------------

    doc = Document(
        filename=file.filename,
        status=status
    )

    db.session.add(doc)
    db.session.commit()

    return redirect(
        url_for("documents.list_documents")
    )


@documents_bp.route("/documents/delete/<int:doc_id>", methods=["POST"])
def delete_document(doc_id):
    """Delete a document from DB, upload folder, FAISS index, and knowledge graph."""

    doc = Document.query.get_or_404(doc_id)
    filename = doc.filename

    # 1. Delete file from upload folder
    file_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        filename
    )
    if os.path.exists(file_path):
        os.remove(file_path)

    # 2. Remove from FAISS index and chunks
    try:
        remove_document(filename)
        reload_retriever()
    except Exception as e:
        print(f"Error removing from FAISS: {e}")

    # 3. Delete from database
    db.session.delete(doc)
    db.session.commit()

    return redirect(url_for("documents.list_documents"))
