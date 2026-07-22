# SAGE--Smart-Asset-Graph-Engine
Turning fragmented plant documents into one queryable knowledge graph — RAG, multi-agent AI, and voice search for industrial ops.
## Structure
```bash
SAGE
|
├── ai_core/                                     
│   ├── dataset_manager.py                              
│   ├── copilot.py                               
│   ├── data_processor.py
│   ├── embeddings.py
│   ├── entity_extractor.py
│   ├── explain_prediction.py
│   ├── ingestion.py
│   ├── knowledge_graph.py
│   ├── llm.py
│   ├── predict.py
│   ├── recommendation.py
│   ├── relationship_extractor.py
│   ├── retriever.py
│   ├── train_model.py
│   └── voice.py
│
├── data/                                         
│   ├── nasa/
│   │   └── archive/
│   │       └── CMaps/
│   ├── uploads/                                  
│   ├── chunks.pkl
│   ├── faiss.index
│   └── metadata.json
│
├── models/                                      
│   ├── rul_model.pkl
│   ├── failure_model.pkl
│   ├── feature_columns.pkl
│   └── feature_importance.pkl
│
├── website/                                                                  
│   ├── db_extension.py                            
│   ├── models.py                                  
│   │
│   ├── routes/
│   │   ├── __init__.py                            
│   │   ├── dashboard.py
│   │   ├── chat.py
│   │   ├── documents.py
│   │   ├── predict.py
│   │   ├── knowledge_graph.py
│   │   ├── voice.py
│   │   └── entities.py
│   │
│   ├── services/
│   │   ├── __init__.py                            
│   │   └── ai_bridge.py                           
│   │
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── chat.html
│   │   ├── documents.html
│   │   ├── predict.html
│   │   ├── knowledge_graph.html
│   │   └── entities.html
│   │
│   └── static/
│       ├── css/
│       │   └── main.css
│       └── images/
│           └── sage_logo.png                       
│
├── notebooks/
│   └── ingestion.ipynb
│
├── instance/                                     
│
├── env/                                           
│
├── knowledge_graph.gexf                            
├── knowledge_graph.graphml
│
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt                                
└── run.py                                          

SAGE transforms scattered engineering artifacts into an intelligent assistant for maintenance and operations. By combining RAG, knowledge graphs, predictive models, and explainable AI, SAGE provides engineering teams with faster diagnostics, richer context, and evidence‑backed recommendations. 