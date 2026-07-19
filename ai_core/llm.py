import logging
from .entity_extractor import extract_entities

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self):

        logger.info("Loading LLM...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-base"
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base"
        )

        logger.info("LLM Loaded.")

    def build_prompt(self, question, context):

        prompt = f"""
    You are an Industrial AI assistant.

    Use ONLY the information provided in the context.

    If the context does not contain the answer,
    reply:
    "I don't know based on the available information."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

        return prompt

    def generate(
        self,
        question,
        context
    ):

        prompt = self.build_prompt(
            question,
            context
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
    **inputs,
    max_new_tokens=150,
    num_beams=5,
    repetition_penalty=1.2,
    length_penalty=1.0,
    early_stopping=True
)

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer

    def ask(
        self,
        question,
        context
    ):

        return self.generate(
            question,
            context
        )


if __name__ == "__main__":

    from .retriever import retrieve

    retriever = retrieve()
    llm = LLMEngine()

    while True:

        question = input("\nAsk: ")

        if question.lower() == "exit":
            break

        context = retriever.retrieve(question)

        answer = llm.ask(
            question,
            context
        )

        print("\nAnswer:")
        print(answer)