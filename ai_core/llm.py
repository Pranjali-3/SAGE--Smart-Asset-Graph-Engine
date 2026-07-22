import logging
import re

from .model_manager import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Loading FLAN-T5 from ModelManager...")
        logger.info("=" * 60)

        self.tokenizer = models.llm_tokenizer
        self.model = models.llm_model

        logger.info("LLM Ready.")

    # ==========================================================
    # Prompt Builder
    # ==========================================================

    def build_prompt(self, question, context):

        return f"""answer the question using the information below.

information: {context}

question: {question}

write a detailed answer in 3 to 5 sentences:"""

    # ==========================================================
    # Remove duplicate sentences
    # ==========================================================

    def clean_context(self, context):

        sentences = re.split(r'(?<=[.!?])\s+', context)

        unique = []
        seen = set()

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 5:
                continue

            if sentence not in seen:
                unique.append(sentence)
                seen.add(sentence)

        return " ".join(unique)

    # ==========================================================
    # Generate Answer
    # ==========================================================

    def generate(self, question, context):

        context = self.clean_context(context)

        prompt = self.build_prompt(
            question,
            context
        )

        inputs = self.tokenizer(

            prompt,

            return_tensors="pt",

            truncation=True,

            max_length=2048

        )

        outputs = self.model.generate(

            **inputs,

            max_new_tokens=300,

            do_sample=True,

            temperature=0.6,

            top_p=0.85,

            no_repeat_ngram_size=3,

            repetition_penalty=1.2,

            length_penalty=1.2

        )

        answer = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )

        # Remove prompt if FLAN echoes it
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        # Safety fallback
        if len(answer.strip()) == 0:
            answer = "I don't know based on the available documents."

        return answer

    # ==========================================================
    # Chat Interface
    # ==========================================================

    def ask(self, question, retriever):

        ranked_results = retriever.retrieve(
            question,
            top_k=5
        )

        # ----------------------------------------
        # Keep top 5 chunks for richer context
        # ----------------------------------------

        ranked_results = ranked_results[:5]

        context_parts = []

        seen_chunks = set()

        sources = []

        seen_sources = set()

        for result in ranked_results:

            chunk = result["chunk"]

            if isinstance(chunk, dict):

                text = chunk.get("text", "")
                source = chunk.get("source", "Unknown")

            else:

                text = str(chunk)
                source = "Unknown"

            if text not in seen_chunks:

                context_parts.append(text)
                seen_chunks.add(text)

            if source not in seen_sources:

                sources.append(source)
                seen_sources.add(source)

        context = "\n\n".join(context_parts)

        answer = self.generate(
            question,
            context
        )

        return {

            "question": question,

            "answer": answer,

            "context": context,

            "sources": sources

        }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    from .retriever import Retriever

    retriever = Retriever()

    llm = LLMEngine()

    print("\nIndustrial AI Assistant")
    print("=" * 70)

    while True:

        question = input("\nAsk : ")

        if question.lower() == "exit":
            break

        response = llm.ask(

            question,

            retriever

        )

        print("\nAnswer")
        print("=" * 70)
        print(response["answer"])

        print("\nSources")
        print("=" * 70)

        for source in response["sources"]:
            print("-", source)