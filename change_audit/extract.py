"""Entity + outcome extraction from evidence text.

Backend "regex"      : rule-based pre-labeller (the same role regex plays in Part II's fine-tuning recipe:
                       "pre-label with regex, hand-correct, fine-tune"). Runs anywhere.
Backend "finetuned"  : the token-classification model produced by fine_tune_change_ner.py
                       (bert-base-cased fine-tuned on data/change_audit/ner_train.jsonl). Needs transformers.

Labels: APP, VERSION, ENV, HOST, TIME, RESULT
"""
from __future__ import annotations
import re

KNOWN_APPS = ["payments-gateway", "customer-portal", "ledger-service", "risk-engine",
              "reporting-hub", "notification-svc", "fx-pricing", "onboarding-api"]

_ENV_CANON = {"prod": "PROD", "production": "PROD", "uat": "UAT", "sit": "SIT", "qa": "QA",
              "dev": "DEV", "preprod": "PREPROD", "pre-prod": "PREPROD"}

_RX = {
    "VERSION": re.compile(r"\bv\d+\.\d+\.\d+\b"),
    "HOST": re.compile(r"\b(?:prod|uat|sit|qa|dev)-[a-z]+-\d{2}\b", re.I),
    "TIME": re.compile(r"\b\d{2}:\d{2} UTC\b"),
    "APP": re.compile(r"\b(?:" + "|".join(map(re.escape, KNOWN_APPS)) + r")\b", re.I),
    "ENV": re.compile(r"\b(?:prod|production|uat|sit|qa|dev|pre-?prod)\b", re.I),
}

# outcome cues -------------------------------------------------------------------------------
PASS_CUES = [
    r"\ball \d+ cases passed", r"\b(\d+) of \1 cases passed\b", r"\ball passed\b", r"\bmatched\b",
    r"\bzero differences\b", r"\bvariance 0\b", r"\bHTTP 200\b", r"\bstatus UP\b", r"\bhealthy\b",
    r"\bno alerts triggered\b", r"\bwithin (?:the )?(?:threshold|baseline)\b", r"\bstatus SUCCESS\b",
    r"\breturn code 0\b", r"\b0 critical, 0 high\b", r"\brestored\b", r"\bsuccessful\b(?= in| at)",
    r"\bconfirming\b", r"\bapproved\b", r"\bresult pass\b", r"\bbelow threshold\b", r"\bsmoke suite result, .*? OK\b",
]
FAIL_CUES = [
    r"\b\d+ of \d+ cases passed, \d+ failed", r"\bfail(?:ed|ure)\b", r"\bHTTP [45]\d\d\b", r"\bvariance [1-9]\d*\b",
    r"\bthreshold breached\b", r"\b[1-9]\d* critical\b", r"\bABEND\b", r"\bdeclined\b",
    r"(?<!no )\balerts triggered\b",
]
METRIC_RX = re.compile(r"\b\d[\d,\.]*\s?(?:ms|percent|%|tps|min|rows)\b", re.I)

_PASS = [re.compile(p, re.I) for p in PASS_CUES]
_FAIL = [re.compile(p, re.I) for p in FAIL_CUES]

_ft_pipe = None


def _regex_entities(text: str) -> list[dict]:
    ents = []
    for label, rx in _RX.items():
        for m in rx.finditer(text):
            ents.append({"label": label, "span": m.group(0), "start": m.start(), "end": m.end()})
    for rx in _PASS:
        for m in rx.finditer(text):
            ents.append({"label": "RESULT", "span": m.group(0), "start": m.start(), "end": m.end()})
    for rx in _FAIL:
        for m in rx.finditer(text):
            ents.append({"label": "RESULT", "span": m.group(0), "start": m.start(), "end": m.end()})
    hosts = [(e["start"], e["end"]) for e in ents if e["label"] == "HOST"]
    ents = [e for e in ents if not (e["label"] == "ENV" and any(a <= e["start"] and e["end"] <= b for a, b in hosts))]
    return sorted(ents, key=lambda e: e["start"])


def _finetuned_entities(text: str, model_dir: str = "./change-ner-finetuned") -> list[dict]:
    global _ft_pipe
    from transformers import pipeline  # imported lazily: not needed for the regex backend
    if _ft_pipe is None:
        _ft_pipe = pipeline("ner", model=model_dir, aggregation_strategy="simple")
    return [{"label": r["entity_group"], "span": r["word"], "start": r["start"], "end": r["end"]}
            for r in _ft_pipe(text[:512])]


def extract_entities(text: str, backend: str = "regex") -> list[dict]:
    return _finetuned_entities(text) if backend == "finetuned" else _regex_entities(text)


def canon_env(span: str) -> str:
    s = span.lower()
    if "-" in s and s.split("-")[0] in _ENV_CANON:       # host name like prod-app-01
        return _ENV_CANON[s.split("-")[0]]
    return _ENV_CANON.get(s.replace(" ", ""), s.upper())


def environments(ents: list[dict]) -> set[str]:
    out = set()
    for e in ents:
        if e["label"] == "ENV":
            out.add(canon_env(e["span"]))
        elif e["label"] == "HOST":
            out.add(canon_env(e["span"]))
    return out


def outcome(text: str, ents: list[dict] | None = None) -> str:
    """'fail' | 'pass' | 'none' (no explicit checkable outcome)."""
    if any(rx.search(text) for rx in _FAIL):
        return "fail"
    if any(rx.search(text) for rx in _PASS) or METRIC_RX.search(text):
        return "pass"
    return "none"
