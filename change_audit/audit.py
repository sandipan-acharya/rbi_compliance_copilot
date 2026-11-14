"""Core audit: does the evidence in a change record support its validation plan?

Pipeline per ticket
 1. classify each validation-plan step into a check type (similarity to the standard's taxonomy clauses)
 2. PLAN_GAP: compare covered check types with those the standard requires for this risk/category (clause metadata)
 3. match every plan step to its best evidence item (semantic similarity: step vs filename + caption)
 4. quality rules on the matched evidence, driven by extracted entities (NER) and clause parameters:
    timing (EV-01), environment (EV-02), outcome (EV-03), scope (EV-04), failed-but-closed (EV-05), completeness (EV-06)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np

from .backends import get_embedder
from .extract import extract_entities, environments, outcome, KNOWN_APPS
from .standard import Standard

DEFAULT_TAU = {"tfidf": 0.20, "minilm": 0.40}     # match threshold; recalibrate per backend (evaluate.py)
DEFAULT_TAU_TYPE = {"tfidf": 0.10, "minilm": 0.25}


@dataclass
class Finding:
    type: str
    clause: str
    severity: str
    step_idx: int | None
    step: str | None
    evidence: str | None
    detail: str
    key: object = None

    def asdict(self):
        return asdict(self)


@dataclass
class StepMatch:
    idx: int
    step: str
    check_type: str | None
    type_score: float
    evidence_idx: int | None
    evidence_file: str | None
    score: float


@dataclass
class AuditResult:
    number: str
    matches: list[StepMatch] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def verdict(self):
        if not self.findings:
            return "Meets standard"
        return "Non-compliant" if any(f.severity == "High" for f in self.findings) else "Needs remediation"


def _stem(fname: str) -> str:
    return re.sub(r"[_\-.]+", " ", re.sub(r"\.\w+$", "", fname))


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class Auditor:
    def __init__(self, backend: str = "tfidf", standard: Standard | None = None, ner_backend: str = "regex",
                 tau: float | None = None, tau_type: float | None = None, fit_texts=None,
                 type_aware: bool = True, type_bonus: float = 0.20):
        self.std = standard or Standard()
        self.emb = get_embedder(backend)
        self.backend, self.ner_backend = backend, ner_backend
        self.type_aware, self.type_bonus = type_aware, type_bonus
        self.tau = DEFAULT_TAU[backend] if tau is None else tau
        self.tau_type = DEFAULT_TAU_TYPE[backend] if tau_type is None else tau_type
        corpus = [f"{c.title}. {c.text}" for c in self.std.clauses] + list(fit_texts or [])
        self.emb.fit(corpus)
        self.std.build_index(self.emb)
        self.taxo = self.std.taxonomy()
        self._tmat = self.emb.encode(list(self.taxo.values()))
        self._tkeys = list(self.taxo)

    # ---------------------------------------------------------------------------------------
    def classify_steps(self, texts, allow_non_validation: bool = False):
        """Map each text to the closest check type in the standard's taxonomy (or None if below threshold)."""
        if not texts:
            return []
        sims = self.emb.encode(texts) @ self._tmat.T
        if not allow_non_validation:
            sims[:, self._tkeys.index("non_validation")] = -1
        out = []
        for row in sims:
            j = int(np.argmax(row))
            out.append((self._tkeys[j] if row[j] >= self.tau_type else None, float(row[j])))
        return out

    def audit(self, rec: dict) -> AuditResult:
        res = AuditResult(rec["number"])
        std, steps, evid = self.std, rec["validation_plan"], rec["evidence"]
        types = self.classify_steps(steps)

        # --- PLAN_GAP (EV: clause metadata) -----------------------------------------------------
        required = std.required_checks(rec)
        covered = {t for t, _ in types if t}
        for chk, clause in required.items():
            if chk not in covered:
                res.findings.append(Finding(
                    "PLAN_GAP", clause, std.severity("PLAN_GAP"), None, None, None,
                    f"Validation plan has no '{chk.replace('_', ' ')}' step although {clause} requires it for a "
                    f"{rec['risk']}-risk {rec['category'].replace('_', ' ')} change.", key=chk))

        # --- step <-> evidence matching ------------------------------------------------------------
        ev_text = [f"{_stem(e['filename'])}. {e['text']}" for e in evid]
        ev_ents = [extract_entities(e["text"], self.ner_backend) for e in evid]
        S = (self.emb.encode(steps) @ self.emb.encode(ev_text).T) if evid and steps else np.zeros((len(steps), 0))
        if self.type_aware and S.shape[1]:
            # stage 2 of matching: classify each evidence item into a check type, drop documents the standard says
            # are not validation evidence (approvals, release notes, runbooks), reward same-type pairs
            ev_types = [t for t, _ in self.classify_steps(ev_text, allow_non_validation=True)]
            for j, et in enumerate(ev_types):
                if et == "non_validation":
                    S[:, j] = -1.0
            for i, (st, _) in enumerate(types):
                for j, et in enumerate(ev_types):
                    if st and st == et:
                        S[i, j] += self.type_bonus
        start, end = _dt(rec["planned_start"]), _dt(rec["planned_end"])
        window = timedelta(hours=std.timing_window_hours(rec))

        for i, step in enumerate(steps):
            ctype, tscore = types[i]
            j = int(np.argmax(S[i])) if S.shape[1] else None
            best = float(S[i, j]) if j is not None else 0.0
            if j is None or best < self.tau:
                res.matches.append(StepMatch(i, step, ctype, tscore, None, None, best))
                res.findings.append(Finding("MISSING_EVIDENCE", "CM-EV-06", std.severity("MISSING_EVIDENCE"), i, step,
                                            None, "No attachment or work note supports this validation step.", key=i))
                continue
            e, ents = evid[j], ev_ents[j]
            res.matches.append(StepMatch(i, step, ctype, tscore, j, e["filename"], best))
            label = f"{e['filename']}: \"{e['text']}\""
            rollback = ctype == "rollback_verification"

            # EV-01 timing
            if not rollback:
                cap = _dt(e["captured_at"])
                if cap < start:
                    res.findings.append(Finding("TIMING", "CM-EV-01", std.severity("TIMING"), i, step, label,
                        f"Evidence captured {e['captured_at']}, before implementation start {rec['planned_start']} "
                        f"(possible reuse of earlier evidence).", key=i))
                elif cap > end + window:
                    res.findings.append(Finding("TIMING", "CM-EV-01", std.severity("TIMING"), i, step, label,
                        f"Evidence captured {e['captured_at']}, more than {int(window.total_seconds() // 3600)}h after "
                        f"implementation end {rec['planned_end']}.", key=i))
            # EV-02 environment
            envs = environments(ents)
            if not rollback and envs and envs != {"PROD"} and not envs & {"PROD"}:
                res.findings.append(Finding("WRONG_ENVIRONMENT", "CM-EV-02", std.severity("WRONG_ENVIRONMENT"), i,
                    step, label, f"Evidence refers to {', '.join(sorted(envs))}; a production change needs production evidence.", key=i))
            # EV-04 scope
            scope_issue = None
            apps = {m.lower() for m in re.findall("|".join(KNOWN_APPS), e["text"], flags=re.I)}
            if apps and rec["application"].lower() not in apps:
                scope_issue = f"names application {', '.join(sorted(apps))} instead of {rec['application']}"
            vers = set(re.findall(r"\bv\d+\.\d+\.\d+\b", e["text"]))
            allowed = {rec["to_version"]} | ({rec["from_version"]} if rollback else set())
            if vers and not vers & allowed:
                scope_issue = f"names version {', '.join(sorted(vers))} but the change deploys {rec['to_version']}"
            if scope_issue:
                res.findings.append(Finding("SCOPE_MISMATCH", "CM-EV-04", std.severity("SCOPE_MISMATCH"), i, step,
                                            label, f"Evidence {scope_issue}.", key=i))
            # EV-03 outcome / EV-05 failed but closed
            oc = outcome(e["text"], ents)
            if oc == "none":
                res.findings.append(Finding("WEAK_EVIDENCE", "CM-EV-03", std.severity("WEAK_EVIDENCE"), i, step, label,
                    "Evidence states no checkable outcome (no count, status, metric or result).", key=i))
            elif oc == "fail" and rec["close_code"] == "Successful":
                res.findings.append(Finding("FAILED_BUT_CLOSED", "CM-EV-05", std.severity("FAILED_BUT_CLOSED"), i, step,
                    label, "Evidence shows a failed result but the change was closed as Successful.", key=i))
        return res


def to_pred_keys(res: AuditResult):
    return {(f.type, f.key) for f in res.findings}
