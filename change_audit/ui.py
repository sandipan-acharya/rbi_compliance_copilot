"""Streamlit UI for the Change Validation Audit Copilot (used as tab 5 of app.py and by app_change_audit.py)."""
from __future__ import annotations
import json, time
from pathlib import Path

import pandas as pd
import streamlit as st

from .audit import Auditor
from .backends import minilm_available
from .recommend import recommend
from .standard import Standard

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "change_audit"
SEV_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
EXAMPLES = ["What evidence is needed for a database change?",
            "Can UAT screenshots be used to validate a production change?",
            "How long after implementation can evidence be captured?",
            "What happens if a validation step failed but the change was closed as successful?"]


@st.cache_data(show_spinner=False)
def _load():
    recs = [json.loads(l) for l in open(DATA / "changes.jsonl")]
    return recs


@st.cache_resource(show_spinner=False)
def _auditor(backend: str):
    recs = _load()
    dev = recs[:60]
    fit = [f"{e['filename']} {e['text']}" for r in dev for e in r["evidence"]] + [s for r in dev for s in r["validation_plan"]]
    kw = {}
    mp = ROOT / "outputs" / f"metrics_{backend}.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        kw = dict(tau=m["tau"], tau_type=m["tau_type"])
    return Auditor(backend, fit_texts=fit, **kw)


@st.cache_data(show_spinner=False)
def _audit_all(backend: str):
    a = _auditor(backend)
    t0 = time.perf_counter()
    res = [a.audit(r) for r in _load()]
    return res, time.perf_counter() - t0


def _llm():
    """Optional SmolLM2 polish (Part I/III model). Returns None when transformers or the model is unavailable."""
    try:
        from transformers import pipeline
        pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct", max_new_tokens=400, do_sample=False)
        return lambda p: pipe(p, return_full_text=False)[0]["generated_text"].strip()
    except Exception:
        return None


def _step_status(result, idx):
    f = [x for x in result.findings if x.step_idx == idx]
    if not f:
        return "✅ OK"
    return " ".join(SEV_ICON[x.severity] for x in f[:1]) + " " + ", ".join(x.type.replace("_", " ").title() for x in f)


def render():
    st.header("Change Validation Audit Copilot")
    st.caption("Situated-learning adaptation of the Part I–III copilot: checks whether the evidence attached to a "
               "change ticket supports its validation plan. **All records are synthetic**; the standard is a "
               "synthetic, generic change-control standard, not any firm's policy.")
    if not (DATA / "changes.jsonl").exists():
        st.error("Run `python -m change_audit.synth` first to generate the synthetic change records.")
        return

    backends = ["tfidf"] + (["minilm"] if minilm_available() else [])
    c1, c2 = st.columns([1, 3])
    backend = c1.selectbox("Similarity backend", backends,
                           help="tfidf runs anywhere. minilm = all-MiniLM-L6-v2, the embedder used by the Part III RAG index.")
    recs = _load()
    results, elapsed = _audit_all(backend)
    std = _auditor(backend).std
    c2.info(f"{len(recs)} synthetic tickets audited in {elapsed:.2f}s on the **{backend}** backend.")

    t_single, t_batch, t_rag = st.tabs(["Audit one ticket", "Annual batch audit", "Ask the standard (RAG)"])

    # ---------------------------------------------------------------------------------------
    with t_single:
        default = next((i for i, r in enumerate(results) if len(r.findings) >= 3 and any(f.severity == "High" for f in r.findings)), 0)
        labels = [f"{r['number']} · {r['application']} · {r['risk']} risk · {r['category'].replace('_', ' ')}" for r in recs]
        i = st.selectbox("Change ticket", range(len(recs)), index=default, format_func=lambda k: labels[k])
        rec, res = recs[i], results[i]
        m = st.columns(5)
        m[0].metric("Type", rec["type"]); m[1].metric("Risk", rec["risk"]); m[2].metric("Close code", rec["close_code"])
        m[3].metric("Findings", len(res.findings)); m[4].metric("Result", res.verdict)
        st.caption(f"{rec['short_description']} · window {rec['planned_start'].replace('T', ' ')} → "
                   f"{rec['planned_end'].replace('T', ' ')} · group {rec['assignment_group']}")

        st.subheader("Validation plan ↔ evidence trace")
        rows = [{"#": mt.idx + 1, "Validation plan step": mt.step, "Check type": (mt.check_type or "?").replace("_", " "),
                 "Matched evidence": mt.evidence_file or "— none —", "Similarity": round(mt.score, 2),
                 "Audit result": _step_status(res, mt.idx)} for mt in res.matches]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with st.expander(f"All attachments on the ticket ({len(rec['evidence'])})"):
            st.dataframe(pd.DataFrame(rec["evidence"]), hide_index=True, use_container_width=True)

        st.subheader("Findings")
        if not res.findings:
            st.success("Evidence matches the validation plan and meets the standard.")
        for f in sorted(res.findings, key=lambda f: ("High", "Medium", "Low").index(f.severity)):
            where = f"Step {f.step_idx + 1}" if f.step_idx is not None else "Plan"
            with st.expander(f"{SEV_ICON[f.severity]} {f.severity} · {f.type.replace('_', ' ').title()} · {where} · {f.clause}", expanded=True):
                st.write(f.detail)
                if f.evidence:
                    st.caption(f"Evidence: {f.evidence}")
                c = std.by_id[f.clause]
                st.caption(f"{c.id} — {c.title}: {c.text}")

        st.subheader("Recommendation to the application team")
        polish = st.checkbox("Polish wording with SmolLM2 (Part I model, optional)", value=False,
                             help="Facts always come from the findings; the LLM only rewrites the wording.")
        llm = _llm() if polish else None
        if polish and llm is None:
            st.warning("transformers / SmolLM2 not available in this environment; showing the grounded template text.")
        note = recommend(rec, res, std, llm=llm)
        st.text_area("Recommendation note (editable, copy into the ticket)", note, height=440, key=f"rec_{rec['number']}_{polish}")

    # ---------------------------------------------------------------------------------------
    with t_batch:
        df = pd.DataFrame([{"Ticket": r["number"], "Application": r["application"], "Risk": r["risk"],
                            "Category": r["category"], "Result": x.verdict,
                            "High": sum(f.severity == "High" for f in x.findings),
                            "Medium": sum(f.severity == "Medium" for f in x.findings),
                            "Low": sum(f.severity == "Low" for f in x.findings),
                            "Finding types": ", ".join(sorted({f.type for f in x.findings}))}
                           for r, x in zip(recs, results)])
        k = st.columns(4)
        k[0].metric("Tickets audited", len(df))
        k[1].metric("Need action", f"{(df['Result'] != 'Meets standard').mean():.0%}")
        k[2].metric("Non-compliant (High)", int((df["Result"] == "Non-compliant").sum()))
        k[3].metric("Total findings", int(df[["High", "Medium", "Low"]].sum().sum()))
        counts = pd.Series([f.type.replace("_", " ").title() for x in results for f in x.findings]).value_counts()
        st.bar_chart(counts, horizontal=True)
        st.caption("Triage queue: non-compliant tickets with the most High findings first.")
        q = df.assign(_o=df["Result"].map({"Non-compliant": 0, "Needs remediation": 1, "Meets standard": 2})) \
              .sort_values(["_o", "High", "Medium"], ascending=[True, False, False]).drop(columns="_o")
        st.dataframe(q, hide_index=True, use_container_width=True, height=330)
        flat = pd.DataFrame([{"ticket": r["number"], "application": r["application"], "group": r["assignment_group"],
                              "severity": f.severity, "finding": f.type, "clause": f.clause,
                              "step": f.step_idx + 1 if f.step_idx is not None else "", "detail": f.detail,
                              "evidence": f.evidence or ""} for r, x in zip(recs, results) for f in x.findings])
        st.download_button("Download all findings (CSV)", flat.to_csv(index=False), "change_audit_findings.csv", "text/csv")
        mp = ROOT / "outputs" / f"metrics_{backend}.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            ho = m["arms"]["full"]["holdout"]
            ev = pd.DataFrame([{"Finding type": t.replace("_", " ").title(), "Precision": round(v["precision"], 2),
                                "Recall": round(v["recall"], 2), "F1": round(v["f1"], 2), "Planted": v["tp"] + v["fn"]}
                               for t, v in ho["per_type"].items()])
            with st.expander("Accuracy against planted defects (fresh synthetic holdout, not used for tuning)"):
                st.dataframe(ev, hide_index=True, use_container_width=True)
                o = ho["overall"]
                st.caption(f"Overall precision {o['precision']:.2f}, recall {o['recall']:.2f}, F1 {o['f1']:.2f}. "
                           "Synthetic data shares vocabulary with the rules, so treat this as an upper bound.")

    # ---------------------------------------------------------------------------------------
    with t_rag:
        st.caption("Same RAG pattern as the Part III chatbot, over the change-validation standard instead of RBI circulars.")
        ex = st.pills("Try an example", EXAMPLES) if hasattr(st, "pills") else None
        q = st.text_input("Ask about the standard", value=ex or "")
        if q:
            out = std.answer(q, k=3)
            st.markdown("**Most relevant clauses**")
            for c, score in out["clauses"]:
                st.markdown(f"**{c.id} — {c.title}**  \n{c.text}")
                st.caption(f"retrieval score {score:.2f}")
