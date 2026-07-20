from .retriever import Retriever
from .llm import LLMEngine


class Copilot:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMEngine()

    def ask(self, question: str, top_k: int = 20, max_chunks: int = 8):
        ranked_results = self.retriever.retrieve(question, top_k=top_k)

        context = self.retriever.get_context(ranked_results, max_chunks=max_chunks)

        answer = self.llm.ask(question, context)

        sources = list({
            result["chunk"]["source"] if isinstance(result["chunk"], dict) else None
            for result in ranked_results[:max_chunks]
        } - {None})

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": ranked_results[:max_chunks]
        }