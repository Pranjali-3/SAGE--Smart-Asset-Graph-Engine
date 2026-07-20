from flask import Blueprint, render_template, request
from website.db_extension import db
from website.models import Session, Message
from website.services.ai_bridge import ask_copilot

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    answer = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        if query:
            result = ask_copilot(query)
            answer = result
            session = Session()
            db.session.add(session)
            db.session.flush()  

            db.session.add(Message(session_id=session.id, role="user", content=query))
            db.session.add(Message(
                session_id=session.id,
                role="assistant",
                content=str(result)
            ))
            db.session.commit()

    return render_template("chat.html", answer=answer)