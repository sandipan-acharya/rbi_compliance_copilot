"""Recommendation note for the application team (text-generation component).

Default: deterministic, clause-grounded templates, so every sentence traces to a finding and a clause.
Optional: pass `llm` (callable prompt->str, e.g. the SmolLM2 pipeline used in Part I textgen.py) to polish
the wording. The facts (ticket, step, clause) are always taken from the findings, never from the LLM.
"""
from __future__ import annotations
from .standard import Standard

ACTION = {
    "MISSING_EVIDENCE": "Attach evidence showing the outcome of this step (captured from production after "
                        "implementation start), or attach a change-manager-approved waiver explaining why it was not performed.",
    "WRONG_ENVIRONMENT": "Re-run the check in production and attach the production result. Lower-environment results "
                         "cannot be used to validate a production change.",
    "TIMING": "Attach evidence captured inside the validation window for this change. If the evidence was reused, "
              "re-capture it; if the capture time metadata is wrong, explain it in a work note.",
    "WEAK_EVIDENCE": "Replace the attachment with one that shows an explicit result (pass count, status code, metric "
                     "value, row count or scan result) and who captured it.",
    "SCOPE_MISMATCH": "Provide evidence for the correct application and release version for this change.",
    "FAILED_BUT_CLOSED": "Re-open the ticket. Close it as Failed or Successful with issues and link the follow-up "
                         "incident or defect, or attach a passing re-test.",
    "PLAN_GAP": "Update the validation plan template for this change type to include the missing step and attach its "
                "evidence, or obtain a documented exception.",
}
ORDER = {"High": 0, "Medium": 1, "Low": 2}


def recommend(rec: dict, result, std: Standard, llm=None) -> str:
    if not result.findings:
        return (f"{rec['number']} ({rec['application']}): validation evidence matches the validation plan and meets "
                f"the standard. No action required.")
    lines = [f"Audit recommendation for {rec['number']} - {rec['application']} ({rec['risk']} risk, "
             f"{rec['category'].replace('_', ' ')}), assignment group {rec['assignment_group']}",
             f"Overall result: {result.verdict}. {len(result.findings)} finding(s).", ""]
    for n, f in enumerate(sorted(result.findings, key=lambda f: (ORDER[f.severity], f.step_idx or 0)), 1):
        where = f"Step {f.step_idx + 1} ('{f.step}')" if f.step_idx is not None else "Validation plan"
        lines.append(f"{n}. [{f.severity}] {f.type.replace('_', ' ').title()} - {where}. Clause {f.clause} "
                     f"({std.by_id[f.clause].title}).")
        lines.append(f"   Observation: {f.detail}")
        if f.evidence:
            lines.append(f"   Evidence reviewed: {f.evidence}")
        lines.append(f"   Recommended action: {ACTION[f.type]}")
    text = "\n".join(lines)
    if llm is not None:
        text = llm("Rewrite the following audit note in clear, polite plain English for an application team. "
                   "Keep every ticket number, step, clause ID and action unchanged.\n\n" + text)
    return text
