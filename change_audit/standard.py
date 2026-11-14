"""Knowledge base: the change validation evidence standard (RAG component).

Clauses carry machine-readable metadata in [key=value] tags, so the audit rules are read FROM the
document rather than hard-coded. Retrieval over clause text supports free-form auditor questions.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "change_audit" / "change_standard.txt"
_HEAD = re.compile(r"^(CM-[A-Z]{2}-\d{2})\s*\|\s*(.+)$")
_TAG = re.compile(r"\[([\w.]+)=([^\]]+)\]")


@dataclass
class Clause:
    id: str
    title: str
    text: str
    tags: dict = field(default_factory=dict)


class Standard:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.clauses: list[Clause] = self._parse(Path(path).read_text(encoding="utf-8"))
        self.by_id = {c.id: c for c in self.clauses}
        self._index = None

    @staticmethod
    def _parse(raw: str) -> list[Clause]:
        out, cur = [], None
        for line in raw.splitlines():
            m = _HEAD.match(line.strip())
            if m:
                head = m.group(2)
                tags = {k: v for k, v in _TAG.findall(head)}
                title = _TAG.sub("", head).replace("|", "").strip()
                cur = Clause(m.group(1), title, "", tags)
                out.append(cur)
            elif cur is not None and line.strip():
                cur.text += (" " if cur.text else "") + line.strip()
        return out

    # ---- rules read from clause metadata --------------------------------------------------
    def required_checks(self, rec: dict) -> dict[str, str]:
        """check_type -> clause id that requires it, for this change record."""
        req = {}
        for c in self.clauses:
            if "requires" not in c.tags:
                continue
            cond = c.tags.get("applies", "all")
            if cond != "all":
                field_, vals = cond.split(":")
                if str(rec.get(field_)) not in vals.split(","):
                    continue
            for chk in c.tags["requires"].split(","):
                req.setdefault(chk, c.id)
        return req

    def timing_window_hours(self, rec: dict) -> int:
        t = self.by_id["CM-EV-01"].tags
        return int(t["emergency_window_hours"] if rec.get("type") == "Emergency" else t["window_hours"])

    def severity(self, finding_type: str) -> str:
        return self.by_id["CM-FD-01"].tags.get(f"sev.{finding_type}", "Medium")

    def taxonomy(self) -> dict[str, str]:
        return {c.tags["check"]: c.text for c in self.clauses if "check" in c.tags}

    # ---- retrieval (RAG) ----------------------------------------------------------------------
    def build_index(self, embedder):
        self._emb = embedder
        texts = [f"{c.title}. {c.text}" for c in self.clauses]
        embedder.fit(texts)
        self._mat = embedder.encode(texts)
        self._index = True

    def retrieve(self, query: str, k: int = 3, exclude_taxonomy: bool = True):
        q = self._emb.encode([query])[0]
        sims = self._mat @ q
        order = np.argsort(-sims)
        res = []
        for i in order:
            c = self.clauses[i]
            if exclude_taxonomy and "check" in c.tags:
                continue
            res.append((c, float(sims[i])))
            if len(res) == k:
                break
        return res

    def answer(self, question: str, k: int = 3, llm=None) -> dict:
        """Extractive answer from retrieved clauses. `llm` is an optional callable(prompt)->str
        (e.g. the SmolLM2 pipeline from Part III rag.py) for a plain-English rewrite."""
        hits = self.retrieve(question, k)
        ctx = "\n".join(f"[{c.id}] {c.title}: {c.text}" for c, _ in hits)
        answer = ctx
        if llm is not None:
            prompt = ("Answer the question using only the clauses below. If the answer is not in them, say "
                      "'Not covered in the standard'.\n\n" + ctx + f"\n\nQuestion: {question}\nAnswer:")
            answer = llm(prompt)
        return {"answer": answer, "sources": [(c.id, round(s, 3)) for c, s in hits], "clauses": hits}
