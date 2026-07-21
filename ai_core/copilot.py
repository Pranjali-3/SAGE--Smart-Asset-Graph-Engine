import logging

from .llm import LLMEngine
from .retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Copilot:

    def __init__(self):

        logger.info("Initializing Copilot...")

        self.retriever = Retriever()

        self.llm = LLMEngine()

        logger.info("Copilot Ready.")

    def ask(self, question):

        return self.llm.ask(
            question,
            self.retriever
        )
