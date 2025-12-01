# Simple RAG Chatbot + Evaluation

This is a lightweight RAG project that demonstrates:

- Vector DB creation
- Retrieval using FAISS
- Simple chatbot function
- RAGAS evaluation (precision, recall, faithfulness)
- Pytest automation

## How to run

### Install dependencies
pip install -r requirements.txt

### Run tests
pytest -v

## Project Files
- src/rag_chatbot.py — Simple RAG chatbot
- src/evaluator.py — RAGAS evaluation
- tests/test_chatbot.py — Automated tests
- data/sample_text.txt — Input text for chatbot
