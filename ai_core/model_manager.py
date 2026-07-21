import logging
import spacy

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class that loads every AI model only once.
    """

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            cls._instance._load_models()

        return cls._instance

    def _load_models(self):

        logger.info("=" * 60)
        logger.info("Loading AI Models...")
        logger.info("=" * 60)

        # ---------------------------------------------------
        # spaCy
        # ---------------------------------------------------

        logger.info("Loading spaCy...")

        self.spacy = spacy.load("en_core_web_sm")

        logger.info("spaCy loaded")

        # ---------------------------------------------------
        # BERT NER
        # ---------------------------------------------------

        logger.info("Loading BERT NER...")

        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            "dslim/bert-base-NER"
        )

        self.bert_model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER"
        )

        self.bert_pipeline = pipeline(
            "ner",
            model=self.bert_model,
            tokenizer=self.bert_tokenizer,
            aggregation_strategy="simple"
        )

        logger.info("BERT loaded")

        # ---------------------------------------------------
        # Embedding Model
        # ---------------------------------------------------

        logger.info("Loading Sentence Transformer...")

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info("Embedding model loaded")

        # ---------------------------------------------------
        # FLAN
        # ---------------------------------------------------

        logger.info("Loading FLAN-T5...")

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-base"
        )

        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base"
        )

        logger.info("FLAN loaded")

        logger.info("=" * 60)
        logger.info("All AI Models Loaded Successfully")
        logger.info("=" * 60)


models = ModelManager()
