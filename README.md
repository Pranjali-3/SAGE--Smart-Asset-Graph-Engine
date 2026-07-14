# SAGE--Smart-Asset-Graph-Engine
Turning fragmented plant documents into one queryable knowledge graph — RAG, multi-agent AI, and voice search for industrial ops.
## Structure
```bash
SAGE
|
|- ai_core/
    |_ copilot.py
    |_ data_processor.py
    |_ embeddings.py
    |_ entity_extraction.py
    |_ ingestion.py
    |_ knowledge_graph.py
    |_ relationship_extractor.py
    |_ retriever.py
    |_ voice.py
|- data/
    |_ nasa/
        |_ archive/
            |_ CMaps/
    |- chunks.pkl
    |- faiss.index
|- notebooks/
    |_ ingestion.ipynb
|- knowledge_graph.gexf
|- knowledge_graph.graphml
```
