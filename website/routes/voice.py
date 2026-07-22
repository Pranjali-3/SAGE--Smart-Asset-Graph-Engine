from flask import Blueprint, request, jsonify
from ai_core.voice import VoiceEngine

voice_bp = Blueprint("voice", __name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = VoiceEngine()
    return _engine


@voice_bp.route("/voice/listen", methods=["POST"])
def listen():
    """
    Record audio from request and return transcribed text.
    Expects JSON with base64-encoded audio.
    """
    try:
        data = request.get_json()
        audio_text = data.get("text")

        if audio_text:
            return jsonify({"success": True, "text": audio_text})

        engine = get_engine()
        text = engine.listen()

        if text:
            return jsonify({"success": True, "text": text})
        else:
            return jsonify({"success": False, "error": "No speech detected"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@voice_bp.route("/voice/speak", methods=["POST"])
def speak():
    """
    Convert text to speech and return audio.
    """
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"success": False, "error": "No text provided"})

        engine = get_engine()
        engine.speak(text)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
