import logging
import speech_recognition as sr
import pyttsx3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceEngine:
    """
    Handles voice input (speech-to-text) and
    voice output (text-to-speech).
    """

    def __init__(self):

        logger.info("Initializing Voice Engine...")

        # ---------------------------------------------------
        # Speech Recognition
        # ---------------------------------------------------

        self.recognizer = sr.Recognizer()

        self.microphone = sr.Microphone()

        # ---------------------------------------------------
        # Text-to-Speech
        # ---------------------------------------------------

        self.tts_engine = pyttsx3.init()

        self._configure_tts()

        # Calibrate once for ambient noise

        with self.microphone as source:

            self.recognizer.adjust_for_ambient_noise(
                source, duration=1
            )

        logger.info("Voice Engine Ready.")

    # ==========================================================
    # TTS Configuration
    # ==========================================================

    def _configure_tts(self):

        voices = self.tts_engine.getProperty("voices")

        # Prefer a female voice if available

        for voice in voices:

            if "female" in voice.name.lower():

                self.tts_engine.setProperty("voice", voice.id)

                break

        self.tts_engine.setProperty("rate", 170)

        self.tts_engine.setProperty("volume", 0.9)

    # ==========================================================
    # Speech-to-Text (Microphone)
    # ==========================================================

    def listen(self, timeout=5, phrase_limit=10):
        """
        Listen from microphone and return text.

        Args:
            timeout: Seconds to wait for speech to start.
            phrase_limit: Max seconds for a phrase.

        Returns:
            str or None: Recognized text, or None on failure.
        """

        try:

            with self.microphone as source:

                logger.info("Listening...")

                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )

            logger.info("Recognizing speech...")

            text = self.recognizer.recognize_google(audio)

            logger.info(f"Recognized: {text}")

            return text

        except sr.WaitTimeoutError:

            logger.warning("No speech detected (timeout).")

            return None

        except sr.UnknownValueError:

            logger.warning("Could not understand audio.")

            return None

        except sr.RequestError as e:

            logger.error(f"Speech recognition error: {e}")

            return None

    # ==========================================================
    # Speech-to-Text (Audio File)
    # ==========================================================

    def transcribe_file(self, file_path):
        """
        Transcribe an audio file to text.

        Args:
            file_path: Path to audio file (WAV, AIFF, FLAC).

        Returns:
            str or None: Transcribed text.
        """

        try:

            with sr.AudioFile(file_path) as source:

                audio = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio)

            logger.info(f"Transcribed: {text}")

            return text

        except Exception as e:

            logger.error(f"Transcription error: {e}")

            return None

    # ==========================================================
    # Text-to-Speech (Speak)
    # ==========================================================

    def speak(self, text):
        """
        Convert text to speech and play it.

        Args:
            text: The text to speak.
        """

        if not text:

            return

        try:

            logger.info(f"Speaking: {text[:80]}...")

            self.tts_engine.say(text)

            self.tts_engine.runAndWait()

        except Exception as e:

            logger.error(f"TTS error: {e}")

    # ==========================================================
    # Text-to-Speech (Save to File)
    # ==========================================================

    def save_speech(self, text, output_path="output.wav"):
        """
        Save spoken text to a WAV file.

        Args:
            text: The text to convert.
            output_path: Path for the output file.
        """

        try:

            self.tts_engine.save_to_file(text, output_path)

            self.tts_engine.runAndWait()

            logger.info(f"Speech saved to {output_path}")

        except Exception as e:

            logger.error(f"Save error: {e}")

    # ==========================================================
    # Continuous Listening Loop
    # ==========================================================

    def continuous_listen(self, callback, wake_word=None):
        """
        Listen continuously and call a function on each phrase.

        Args:
            callback: Function to call with recognized text.
            wake_word: Optional phrase to trigger processing.
        """

        logger.info("Starting continuous listening...")

        while True:

            text = self.listen(timeout=None, phrase_limit=10)

            if text is None:

                continue

            if wake_word and wake_word.lower() not in text.lower():

                continue

            callback(text)


# ==========================================================
# Main (Testing)
# ==========================================================

if __name__ == "__main__":

    engine = VoiceEngine()

    print("\nVoice Engine Test")
    print("=" * 70)

    while True:

        print("\n1. Listen from microphone")
        print("2. Speak text")
        print("3. Transcribe audio file")
        print("4. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":

            print("Speak now...")

            text = engine.listen()

            if text:

                print(f"Recognized: {text}")

            else:

                print("No speech detected.")

        elif choice == "2":

            text = input("Enter text to speak: ")

            engine.speak(text)

        elif choice == "3":

            path = input("Enter audio file path: ")

            text = engine.transcribe_file(path)

            if text:

                print(f"Transcribed: {text}")

        elif choice == "4":

            break

        else:

            print("Invalid choice.")
