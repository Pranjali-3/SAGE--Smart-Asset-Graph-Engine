import logging
from .model_manager import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self):

        logger.info("Loading FLAN-T5 from ModelManager...")

        self.tokenizer = models.llm_tokenizer

        self.model = models.llm_model

        logger.info("LLM Ready.")

    # ==========================================================
    # Prompt Builder (Short for FLAN)
    # ==========================================================

    def build_prompt(self, question, context):

        return f"""Context:

{context}

Question:
{question}

Answer ONLY using the context.
If the answer is missing, say:
I don't know based on the available documents.

Answer:"""

    # ==========================================================
    # Generate Answer
    # ==========================================================

    def generate(self, question, context):

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

            max_new_tokens=120,

            do_sample=True,

            temperature=0.3,

            top_p=0.9

        )

        answer = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )

        return answer

    # ==========================================================
    # Chat Interface (top_k=5, max_chunks=3)
    # ==========================================================

    def ask(self, question, retriever):

        ranked_results = retriever.retrieve(
            question,
            top_k=5
        )

        # Deduplicate chunks
        unique = []
        seen = set()

        for result in ranked_results:

            chunk = result["chunk"]

            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk)

            if text not in seen:
                seen.add(text)
                unique.append(text)

        context = "\n\n".join(unique)

        answer = self.generate(
            question,
            context
        )

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": [
                chunk["chunk"]["source"]
                for chunk in ranked_results
                if isinstance(chunk["chunk"], dict)
            ]
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

        for source in sorted(set(response["sources"])):
            print("-", source)