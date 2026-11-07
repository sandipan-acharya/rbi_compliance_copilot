# RBI KYC & PSL Compliance Copilot

Follows the Assignment Plan exactly.

## Setup
1. Unzip raw_pdfs.zip into data/raw_pdfs/
2. Unzip domain_corpus.zip into data/corpus/
3. pip install -r requirements.txt
4. python -c "from ingest import extract_pdfs; extract_pdfs()"
5. python -c "from rag import build_index; build_index()"
6. streamlit run app.py

## Tabs
- Clause Explorer (NER)
- Explain & Checklist (Text Generation)
- Ask a Clause (Extractive QA)
- Compliance Chatbot (RAG)