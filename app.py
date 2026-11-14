import streamlit as st

st.set_page_config(page_title="RBI KYC & PSL Compliance Copilot", layout="wide")
st.title("RBI KYC & PSL Compliance Copilot")

# Try imports but do not stop the app
try:
    from ner import extract_entities
    from textgen import generate
    from qa import answer_from_clause
    from rag import rag_answer
    st.success("All modules loaded successfully")
    modules_ok = True
except Exception as e:
    st.error(f"Import error: {type(e).__name__}: {e}")
    st.warning("Tabs are shown below, but some features will not work until the import error is fixed.")
    modules_ok = False

# Always create the tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Clause Explorer (NER)",
    "Explain & Checklist",
    "Ask a Clause (QA)",
    "Compliance Chatbot (RAG)"
])

with tab1:
    st.header("Clause Explorer – Named Entity Recognition")
    if not modules_ok:
        st.info("NER module failed to load. Fix the import error above.")
    clause = st.text_area("Paste a clause or page", height=200, key="ner_input")
    if st.button("Extract Entities") and clause and modules_ok:
        with st.spinner("Retrieving and generating answer..."):
            ents = extract_entities(clause)
        for e in ents:
            st.write(f"**{e['label']}**: {e['span']} ({e['score']})")

with tab2:
    st.header("Explain & Checklist")
    if not modules_ok:
        st.info("Text generation module failed to load.")
    text = st.text_area("Clause / scenario", height=180, key="gen_input")
    mode = st.radio("Mode", ["summary", "checklist", "explain"])
    if st.button("Generate") and text and modules_ok:
        with st.spinner("Retrieving and generating answer..."):
            ents = extract_entities(text)
            reply = generate(text, mode=mode, entities=ents)
        st.success(reply)

with tab3:
    st.header("Ask a Clause – Extractive QA")
    if not modules_ok:
        st.info("QA module failed to load.")
    clause = st.text_area("Select clause", height=160, key="qa_input")
    q = st.text_input("Question about this clause")
    if st.button("Answer") and clause and q and modules_ok:
        with st.spinner("Retrieving and generating answer..."):
            res = answer_from_clause(clause, q)
        st.write(f"**Answer**: {res['answer']}")
        st.caption(f"Confidence: {res['score']} | Span: {res['span']}")

with tab4:
    st.header("Compliance Chatbot (RAG)")
    if not modules_ok:
        st.info("RAG module failed to load.")
    entity_filter = st.selectbox("Entity type filter (optional)", 
                                 ["None", "Commercial Bank", "NBFC", "SFB", "AIFI", "ARC"])
    question = st.text_input("Ask across all directions")
    if st.button("Ask RAG") and question and modules_ok:
        et = None if entity_filter == "None" else entity_filter
        with st.spinner("Retrieving and generating answer..."):
            res = rag_answer(question, entity_type=et)
        
        st.subheader("Answer")
        st.write(res.get("answer", "No answer returned"))
        
        st.subheader("Sources")
        sources = res.get("sources", [])
        if sources:
            for s in sources:
                st.write(f"- {s.get('source', 'unknown')} (page: {s.get('page', '?')})")
        else:
            st.write("No sources found")