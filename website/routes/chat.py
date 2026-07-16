from flask import Blueprint, render_template, request
from ..db_extension import db
from ..models import Session, Message
from ..services.ai_bridge import search_knowledge

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    answer = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        if query:
            result = search_knowledge(query)
            answer = result

            # Save to DB (basic: one session per request for now)
            session = Session()
            db.session.add(session)
            db.session.flush()  # get session.id before commit

            db.session.add(Message(session_id=session.id, role="user", content=query))
            db.session.add(Message(
                session_id=session.id,
                role="assistant",
                content=str(result)
            ))
            db.session.commit()

    return render_template("chat.html", answer=answer)