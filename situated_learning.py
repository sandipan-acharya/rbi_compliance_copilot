"""Situated-learning demos for SOP support and technical-document research.

Retrieval and summaries are deliberately extractive and local: answers show the
source text they rely on rather than inventing company policy.
"""
from __future__ import annotations

from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_DIR = Path(__file__).resolve().parent / "data" / "situated_learning"


def _load_documents(filename: str) -> list[dict[str, str]]:
    path = DATA_DIR / filename
    with path.open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    for row in documents:
        if not isinstance(row.get("source"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"{path} rows must contain string 'source' and 'text' fields.")
    if not documents:
        raise ValueError(f"{path} contains no documents.")
    return documents


SOP_GUIDES = _load_documents("sop_documents.jsonl")
TECHNICAL_GUIDES = _load_documents("technical_documents.jsonl")

ENTITY_PATTERNS = [
    ("Deadline", re.compile(
        r"\b(?:by|before|within|no later than|due|deadline|at least)\b[^.;,\n]{0,80}"
        r"|\b\d{1,2}\s+(?:hours?|days?|weeks?|months?)\b"
        r"|\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b", re.I)),
    ("Required rule", re.compile(
        r"\b[^.;\n]{0,100}\b(?:must|required|mandatory|shall|needs? to)\b[^.;\n]{0,100}", re.I)),
    ("Role", re.compile(
        r"\b(?:administrator|admin|owner|manager|developer|viewer|operator|approver|"
        r"service desk|application owner|business owner)\b", re.I)),
    ("Access / permission", re.compile(
        r"\b(?:access|permission|authorization|authori[sz]ation|role-based|least-privilege|"
        r"denied|restricted|approval)\b", re.I)),
    ("Identifier / field", re.compile(
        r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|HTTP\s[1-5]\d\d|/[A-Za-z0-9_./{}-]+")),
]

_REQUIREMENT = re.compile(
    r"\b(?:must|required|mandatory|shall|needs? to|due|deadline|no later than|within)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _read_uploads(files) -> tuple[list[dict[str, str]], list[str]]:
    documents, errors = [], []
    for uploaded in files or []:
        name = uploaded.name
        suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        try:
            if suffix in {"txt", "md"}:
                text = uploaded.getvalue().decode("utf-8-sig")
                if text.strip():
                    documents.append({"source": name, "text": text})
                else:
                    errors.append(f"{name}: file is empty.")
            elif suffix == "pdf":
                from pypdf import PdfReader

                reader = PdfReader(BytesIO(uploaded.getvalue()))
                extracted = 0
                for page_number, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append({
                            "source": f"{name} · page {page_number}",
                            "text": text,
                        })
                        extracted += 1
                if not extracted:
                    errors.append(f"{name}: no extractable text found (scanned PDFs need OCR).")
            else:
                errors.append(f"{name}: unsupported file type; upload PDF, TXT, or Markdown.")
        except Exception as exc:
            errors.append(f"{name}: could not read file ({type(exc).__name__}: {exc}).")
    return documents, errors


def _retrieval_chunks(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    chunks = []
    for document in documents:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document["text"]) if p.strip()]
        if not paragraphs and document["text"].strip():
            paragraphs = [document["text"].strip()]
        for paragraph in paragraphs:
            sentences = [s.strip() for s in _SENTENCE.split(paragraph) if s.strip()]
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) > 1000 and current:
                    chunks.append({"source": document["source"], "text": current})
                    current = ""
                current = f"{current} {sentence}".strip()
            if current:
                chunks.append({"source": document["source"], "text": current})
    return chunks


def _retrieve(query: str, documents: list[dict[str, str]], limit: int = 3):
    chunks = _retrieval_chunks(documents)
    if not query.strip() or not chunks:
        return []
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, lowercase=True,
    )
    matrix = vectorizer.fit_transform([item["text"] for item in chunks])
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    if not len(scores) or scores.max() <= 0:
        return []
    ranked = sorted(range(len(chunks)), key=lambda idx: (-scores[idx], idx))
    return [(chunks[idx], float(scores[idx])) for idx in ranked[:limit]]


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE.split(text.strip()) if sentence.strip()]


def _summary(text: str) -> str:
    parts = _sentences(text)
    if not parts:
        return "Add a section of text to generate a summary."
    return " ".join(parts[:3])


def _checklist(text: str) -> list[str]:
    parts = _sentences(text)
    matches = [part for part in parts if _REQUIREMENT.search(part)]
    return matches or parts[:5]


def _highlighted(text: str) -> str:
    matches = []
    for label, pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), label))

    events = {}
    for start, finish, label in matches:
        events.setdefault(start, []).append((label, 1))
        events.setdefault(finish, []).append((label, -1))

    points = sorted(events)
    active, segments = {}, []
    for index, start in enumerate(points[:-1]):
        for label, delta in events[start]:
            count = active.get(label, 0) + delta
            if count:
                active[label] = count
            else:
                active.pop(label, None)
        finish = points[index + 1]
        if not active or start == finish:
            continue
        labels = ", ".join(sorted(active))
        if segments and segments[-1][1] == start and segments[-1][2] == labels:
            segments[-1] = (segments[-1][0], finish, labels)
        else:
            segments.append((start, finish, labels))

    rendered, pos = [], 0
    for start, finish, labels in segments:
        rendered.append(escape(text[pos:start]))
        rendered.append(
            f'<mark title="{escape(labels)}">{escape(text[start:finish])}</mark>'
            f'<small> [{escape(labels)}]</small>'
        )
        pos = finish
    rendered.append(escape(text[pos:]))
    return (
        '<div style="white-space:pre-wrap;line-height:1.7">'
        + "".join(rendered)
        + "</div>"
    )


def _show_sources(hits):
    if not hits:
        st.info("No relevant source text was retrieved. Add documents or try a more specific query.")
        return
    for item, score in hits:
        with st.expander(f"{item['source']} · relevance {score:.2f}"):
            st.write(item["text"])


def _show_grounded_answer(hits):
    if not hits:
        _show_sources(hits)
        return
    item, score = hits[0]
    st.write(item["text"])
    st.caption(f"Retrieved passage · {item['source']} · relevance {score:.2f}")
    if len(hits) > 1:
        st.markdown("**Other relevant passages**")
        _show_sources(hits[1:])


def render_sop_assistant():
    st.header("Workplace SOP & To-do Assistant")
    st.caption(
        "Find the relevant steps for common access and service-desk issues. "
        "Built-in SOPs are illustrative examples, not any organization's approved policy."
    )
    uploads = st.file_uploader(
        "Add your approved SOPs (PDF, TXT, or Markdown)",
        type=["pdf", "txt", "md"], accept_multiple_files=True, key="sop_uploads",
    )
    uploaded_docs, errors = _read_uploads(uploads)
    for error in errors:
        st.warning(error)
    docs = SOP_GUIDES + uploaded_docs
    issue = st.text_area(
        "Describe the access restriction, to-do item, or workplace problem",
        height=120, key="sop_issue",
    )
    if st.button("Find relevant guidance", key="sop_find"):
        if not issue.strip():
            st.warning("Describe the problem to search the SOPs.")
        else:
            hits = _retrieve(issue, docs)
            st.subheader("Most relevant SOP passage")
            _show_grounded_answer(hits)
            if hits:
                st.caption(
                    "Check the cited approved SOP for the full context. Do not bypass an access restriction."
                )
    with st.expander("SOP sources available in this session"):
        for doc in docs:
            st.caption(doc["source"])


def render_document_workbench():
    st.header("Technical Document Research Workbench")
    st.caption(
        "Extract requirements from a section, create a quick summary or checklist, ask a section-level "
        "question, and search across sources. Retrieval is local and provides source references."
    )
    uploads = st.file_uploader(
        "Upload technical documents (PDF, TXT, or Markdown)",
        type=["pdf", "txt", "md"], accept_multiple_files=True, key="technical_uploads",
    )
    uploaded_docs, errors = _read_uploads(uploads)
    for error in errors:
        st.warning(error)
    docs = TECHNICAL_GUIDES + uploaded_docs

    section = st.text_area(
        "Paste a section to analyze or question", height=180, key="document_section",
    )
    if section.strip():
        st.subheader("Highlighted requirements and entities")
        st.markdown(_highlighted(section), unsafe_allow_html=True)
        st.subheader("Plain-English summary")
        st.write(_summary(section))
        st.subheader("Implementation / validation checklist")
        for item in _checklist(section):
            st.markdown(f"- {item}")

        section_question = st.text_input(
            "Ask a precise question about the pasted section", key="section_question",
        )
        if st.button("Answer from this section", key="section_answer"):
            if not section_question.strip():
                st.warning("Enter a question about the pasted section.")
            else:
                hits = _retrieve(
                    section_question, [{"source": "Pasted section", "text": section}], limit=1,
                )
                st.subheader("Section-grounded answer")
                _show_grounded_answer(hits)

    st.subheader("Ask across all documents")
    query = st.text_input("Search question", key="all_docs_query")
    if st.button("Search documents", key="all_docs_search"):
        if not query.strip():
            st.warning("Enter a question or search phrase.")
        else:
            corpus = docs + (
                [{"source": "Pasted section", "text": section}]
                if section.strip() else []
            )
            st.subheader("Retrieved source passages")
            _show_grounded_answer(_retrieve(query, corpus, limit=4))
