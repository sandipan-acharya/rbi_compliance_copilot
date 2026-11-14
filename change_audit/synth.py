"""Synthetic ServiceNow-style change records with SEEDED, LABELLED defects.

No real workplace data is used. Prevalence of each defect is a knob chosen to make evaluation meaningful;
it says nothing about real defect rates. Generated outputs (data/change_audit/):
  changes.jsonl       change records (what the auditor sees)
  ground_truth.jsonl  planted findings per ticket (what the evaluator knows)
  ner_train.jsonl     BIO-tagged evidence sentences for fine-tuning the entity model (Part II style)
"""
from __future__ import annotations
import json, random, re
from datetime import datetime, timedelta
from pathlib import Path

from .standard import Standard
from .extract import KNOWN_APPS

OUT = Path(__file__).resolve().parent.parent / "data" / "change_audit"
PERSONS = ["A. Rao", "S. Iyer", "M. Khan", "P. Nair", "R. Das", "T. Mehta", "L. Fernandes", "K. Gill"]
GROUPS = ["Payments Platform", "Digital Channels", "Core Ledger", "Risk Technology", "Data Services"]
LOWER_ENVS = ["UAT", "SIT", "DEV", "QA"]

# Template markup:  {slot} -> labelled by SLOT_LABEL ;  [[text|RESULT]] -> labelled span
SLOT_LABEL = {"app": "APP", "ver": "VERSION", "h1": "HOST", "h2": "HOST", "ts": "TIME", "env": "ENV", "prev": "VERSION"}

CHECKS = {
    "smoke_test": dict(
        plan=["Run post-deployment smoke test covering login and primary transaction flow",
              "Execute smoke test suite after release and confirm critical user journeys work",
              "Perform functional smoke test of core screens after deployment"],
        ok=["Smoke test report {app} {ver}: [[all {n} cases passed|RESULT]]",
            "Test run summary, critical journeys executed, [[{n} of {n} cases passed|RESULT]], run by {person}",
            "Screenshot of smoke suite result, login and payment journey [[all passed|RESULT]]"],
        fail=["Smoke test report {app} {ver}: [[22 of 25 cases passed, 3 failed|RESULT]] on payment journey"],
        files=["smoke_test_results.png", "post_deploy_smoke_{num}.pdf", "functional_test_summary.xlsx"]),
    "health_check": dict(
        plan=["Verify application health endpoint returns HTTP 200 on all nodes",
              "Confirm service health check is green on every node",
              "Check health status is UP across all instances"],
        ok=["curl /health [[HTTP 200|RESULT]] on {h1}, {h2} at {ts}",
            "Load balancer view, all nodes [[healthy|RESULT]] for {app} {ver}",
            "Actuator health [[status UP|RESULT]] on {h1} and {h2}"],
        fail=["curl /health [[HTTP 503|RESULT]] on {h2} at {ts}"],
        files=["healthcheck_nodes.png", "lb_status_{num}.png", "actuator_health.txt"]),
    "data_reconciliation": dict(
        plan=["Reconcile row counts between source and migrated tables",
              "Validate data migration with record count and checksum comparison",
              "Confirm no data loss by comparing record totals before and after the change"],
        ok=["Row count query output for {app} {ver}: source 1,204,331 target 1,204,331 [[matched|RESULT]]",
            "Checksum comparison report, [[zero differences|RESULT]]",
            "Pre and post totals reconciled, [[variance 0|RESULT]]"],
        fail=["Row count query output: source 1,204,331 target 1,204,117, [[variance 214|RESULT]] rows"],
        files=["recon_rowcounts.xlsx", "migration_checksum_report.csv", "prepost_totals.png"]),
    "monitoring_check": dict(
        plan=["Monitor error rate and latency dashboards for 30 minutes after the change",
              "Review monitoring dashboard for error spikes after implementation",
              "Observe application alerts and error graphs during the first hour"],
        ok=["Dashboard screenshot, error rate 0.02 percent, p95 latency 180 ms over 30 min for {app}",
            "[[No alerts triggered|RESULT]] in the hour after the change, error graph flat",
            "Monitoring export for {app} showing error rate [[below threshold|RESULT]] from {ts}"],
        fail=["Dashboard: error rate 4.8 percent, [[alerts triggered|RESULT]] at {ts}"],
        files=["grafana_error_rate.png", "monitoring_30min.png", "alerts_review.pdf"]),
    "rollback_verification": dict(
        plan=["Confirm backout procedure was validated in pre-production",
              "Verify rollback steps were tested successfully before implementation",
              "Demonstrate that the backout plan can restore the previous version"],
        ok=["Rollback test executed, previous version {prev} [[restored|RESULT]] in 6 min",
            "Backout rehearsal log for {app}, restore to {prev} [[restored|RESULT]]",
            "Rollback dry run record, [[result pass|RESULT]] for {prev}"],
        fail=["Rollback rehearsal [[failed|RESULT]] at step 4, previous version {prev} not restored"],
        files=["rollback_rehearsal.log", "backout_test_{num}.docx"]),
    "performance_check": dict(
        plan=["Run load test and confirm response time is within the agreed baseline",
              "Validate performance against baseline after deployment",
              "Measure transaction throughput and compare with baseline"],
        ok=["Load test result for {app} {ver}: p95 210 ms against baseline 250 ms, [[within baseline|RESULT]]",
            "Performance report, throughput 480 tps versus baseline 450 tps, [[result pass|RESULT]]",
            "Load test summary, response time 190 ms, [[within threshold|RESULT]]"],
        fail=["Load test result: p95 620 ms against baseline 250 ms, [[threshold breached|RESULT]]"],
        files=["loadtest_summary.pdf", "perf_baseline_compare.xlsx"]),
    "security_check": dict(
        plan=["Run vulnerability scan on the updated component and review findings",
              "Verify TLS certificate and security configuration after the change",
              "Confirm security scan shows no critical or high findings"],
        ok=["Scan report for {app} {ver}: [[0 critical, 0 high|RESULT]] findings",
            "SSL certificate chain verified for {app}, TLS 1.2 or above, [[result pass|RESULT]]",
            "Vulnerability scan summary, [[0 critical, 0 high|RESULT]]"],
        fail=["Scan report for {app} {ver}: [[2 critical|RESULT]], 5 high findings"],
        files=["vuln_scan_report.pdf", "tls_cert_check.txt"]),
    "batch_job_check": dict(
        plan=["Confirm scheduled batch job completes successfully in the next cycle",
              "Verify nightly job run finished without errors after the change",
              "Check the scheduler shows the job ended with success status"],
        ok=["Scheduler view, job ended [[status SUCCESS|RESULT]] at {ts}",
            "Batch log for {app}, [[return code 0|RESULT]], 0 records rejected",
            "Job run record, nightly cycle completed, [[status SUCCESS|RESULT]]"],
        fail=["Scheduler view, job ended with [[ABEND|RESULT]] at {ts}"],
        files=["scheduler_job_status.png", "batch_run_log.txt"]),
    "uat_signoff": dict(
        plan=["Obtain business owner sign-off on post-implementation validation",
              "Get application owner confirmation that the change works as intended",
              "Record owner approval of the validation outcome"],
        ok=["Email from {person}, business owner, [[confirming|RESULT]] the change works as expected at {ts}",
            "Sign-off recorded by application owner {person}, [[approved|RESULT]]",
            "Approval message from {person} for {app} {ver}, [[approved|RESULT]]"],
        fail=["Reply from {person}: owner [[declined|RESULT]] sign-off pending fix"],
        files=["owner_signoff.msg", "approval_confirmation.eml"]),
}
WEAK = ["Tested OK", "Checked, looks fine", "Validation completed successfully", "Done as per plan", "All good", "Verified"]
CATEGORIES = {"application_release": .45, "database_change": .2, "security_config": .15, "batch_schedule": .2}
RISKS = {"Low": .35, "Medium": .4, "High": .25}
P = dict(missing=.05, wrong_env=.03, timing=.04, weak=.04, scope=.03, failed_closed=.02, genuine_fail=.02, plan_gap=.12)

_SLOT = re.compile(r"\{(\w+)\}")
_MARK = re.compile(r"\[\[(.+?)\|(\w+)\]\]")


def _pick(rng, d):
    return rng.choices(list(d), weights=list(d.values()))[0]


def render(template: str, vals: dict):
    """Fill slots and return (text, [(start, end, label)]) with exact character spans."""
    text, spans, pos = "", [], 0
    parts = []
    for m in _MARK.finditer(template):
        parts.append((template[pos:m.start()], None))
        parts.append((m.group(1), m.group(2)))
        pos = m.end()
    parts.append((template[pos:], None))
    for chunk, mark in parts:
        if mark:
            filled = _SLOT.sub(lambda m: str(vals.get(m.group(1), m.group(0))), chunk)
            spans.append((len(text), len(text) + len(filled), mark))
            text += filled
        else:
            last = 0
            for m in _SLOT.finditer(chunk):
                text += chunk[last:m.start()]
                v = str(vals.get(m.group(1), m.group(0)))
                if m.group(1) in SLOT_LABEL:
                    spans.append((len(text), len(text) + len(v), SLOT_LABEL[m.group(1)]))
                text += v
                last = m.end()
            text += chunk[last:]
    return text, spans


def _env_suffix(rng, env, force=False):
    """Optionally name the environment, as captions sometimes do. Returns (text, spans-relative)."""
    opts = [("", None), (f" [{env}]", "ENV"), (f" captured from {env.lower()}-app-01", "HOST")]
    w = [0, .5, .5] if force else [.3, .35, .35]
    txt, lab = rng.choices(opts, weights=w)[0]
    if not lab:
        return "", []
    if lab == "ENV":
        return txt, [(2, 2 + len(env), "ENV")]
    host = f"{env.lower()}-app-01"
    i = txt.index(host)
    return txt, [(i, i + len(host), "HOST")]


def make_evidence(rng, rec, step_idx, check, kind, ts, env="PROD", force_env=False, scope_wrong=None):
    spec = CHECKS[check]
    app, ver, prev = rec["application"], rec["to_version"], rec["from_version"]
    if scope_wrong:
        if scope_wrong[0] == "app":
            app = scope_wrong[1]
        else:
            ver = scope_wrong[1]
    vals = dict(app=app, ver=ver, prev=prev, n=rng.choice([18, 24, 32]), person=rng.choice(PERSONS),
                h1=f"{env.lower()}-app-01", h2=f"{env.lower()}-app-02", ts=ts.strftime("%H:%M UTC"))
    if kind == "weak":
        text, spans = rng.choice(WEAK), []
    else:
        pool = spec["fail"] if kind == "fail" else spec["ok"]
        if scope_wrong:   # need a template that actually names app/version
            tag = "{app}" if scope_wrong[0] == "app" else "{ver}"
            pool = [t for t in pool if tag in t] or [t for t in spec["ok"] if "{app}" in t or "{ver}" in t]
        text, spans = render(rng.choice(pool), vals)
        if check != "rollback_verification":
            suffix, sspans = _env_suffix(rng, env, force=force_env)
            off = len(text)
            text += suffix
            spans += [(a + off, b + off, l) for a, b, l in sspans]
        else:   # rehearsal happens in a lower environment, legitimately
            sfx = f" in {rec['_lower_env']}"
            spans.append((len(text) + 4, len(text) + 4 + len(rec["_lower_env"]), "ENV"))
            text += sfx
    fname = rng.choice(spec["files"]).format(num=rec["number"][-4:])
    return {"filename": fname, "captured_at": ts.isoformat(timespec="minutes"), "text": text,
            "_spans": spans}


def generate(n: int = 200, seed: int = 7):
    rng = random.Random(seed)
    std = Standard()
    recs, truth = [], []
    for i in range(n):
        number = f"CHG{3000000 + i + 1}"
        app = rng.choice(KNOWN_APPS)
        cat, risk = _pick(rng, CATEGORIES), _pick(rng, RISKS)
        ctype = "Emergency" if rng.random() < .15 else "Normal"
        major, minor, patch = rng.randint(1, 4), rng.randint(1, 20), rng.randint(0, 9)
        to_v, from_v = f"v{major}.{minor}.{patch + 1}", f"v{major}.{minor}.{patch}"
        start = datetime(2026, 1, 5) + timedelta(days=rng.randint(0, 270), hours=rng.choice([20, 21, 22, 23, 1, 2]),
                                                minutes=rng.choice([0, 15, 30]))
        end = start + timedelta(minutes=rng.choice([60, 90, 120, 180]))
        rec = dict(number=number, application=app, type=ctype, risk=risk, category=cat, environment="PROD",
                   assignment_group=rng.choice(GROUPS), from_version=from_v, to_version=to_v,
                   planned_start=start.isoformat(timespec="minutes"), planned_end=end.isoformat(timespec="minutes"),
                   short_description=f"{cat.replace('_', ' ').title()} for {app} {from_v} to {to_v}",
                   implementation_plan=f"Deploy {app} {to_v} to production nodes in rolling fashion.",
                   backout_plan=f"Redeploy {app} {from_v} from artefact repository.")
        rec["_lower_env"] = rng.choice(LOWER_ENVS)
        required = std.required_checks(rec)
        plan_checks = list(required)
        gt = []
        if len(plan_checks) >= 3 and rng.random() < P["plan_gap"]:
            drop = rng.choice(plan_checks)
            plan_checks.remove(drop)
            gt.append(("PLAN_GAP", drop))
        extras = [c for c in CHECKS if c not in plan_checks and c not in required and c != "rollback_verification"]
        if rng.random() < .3 and extras:
            plan_checks.append(rng.choice(extras))
        # keep a stable "natural" order
        order = list(CHECKS)
        plan_checks.sort(key=order.index)
        plan = [rng.choice(CHECKS[c]["plan"]) for c in plan_checks]

        # per-step defect draw (one defect per step) -------------------------------------------
        defects = {}
        for idx, c in enumerate(plan_checks):
            r, acc = rng.random(), 0.0
            for d in ["missing", "wrong_env", "timing", "weak", "scope", "failed_closed", "genuine_fail"]:
                acc += P[d]
                if r < acc:
                    if c == "rollback_verification" and d in ("wrong_env", "timing"):
                        break   # not applicable
                    defects[idx] = d
                    break
        genuine = any(d == "genuine_fail" for d in defects.values())
        if genuine:   # a record that honestly failed validation cannot also be a 'closed as success' defect
            defects = {k: v for k, v in defects.items() if v != "failed_closed"}
        close = "Successful"
        if genuine:
            close = rng.choice(["Failed", "Successful with issues"])

        evid = []
        for idx, c in enumerate(plan_checks):
            d = defects.get(idx)
            if d == "missing":
                gt.append(("MISSING_EVIDENCE", idx))
                continue
            # normal capture time
            if c == "rollback_verification":
                ts = start - timedelta(days=rng.randint(1, 5), hours=rng.randint(0, 6))
            elif ctype == "Emergency" and rng.random() < .15:
                ts = end + timedelta(hours=rng.randint(26, 44))
            else:
                ts = start + timedelta(minutes=rng.randint(30, 60 + int((end - start).total_seconds() // 60) + 150))
            kind, env, force, scope = "ok", "PROD", False, None
            if d == "wrong_env":
                env, force = rng.choice(LOWER_ENVS), True
                gt.append(("WRONG_ENVIRONMENT", idx))
            elif d == "timing":
                if rng.random() < .7:
                    ts = start - timedelta(days=rng.randint(1, 3), hours=rng.randint(0, 5))
                else:
                    ts = end + timedelta(hours=rng.randint(60, 90))
                gt.append(("TIMING", idx))
            elif d == "weak":
                kind = "weak"
                gt.append(("WEAK_EVIDENCE", idx))
            elif d == "scope":
                can_app = any("{app}" in t for t in CHECKS[c]["ok"])
                can_ver = any("{ver}" in t for t in CHECKS[c]["ok"])
                use_app = can_app and (not can_ver or rng.random() < .5)
                if use_app:
                    scope = ("app", rng.choice([a for a in KNOWN_APPS if a != app]))
                else:
                    scope = ("ver", f"v{major}.{minor + 2}.{patch + 1}")
                gt.append(("SCOPE_MISMATCH", idx))
            elif d in ("failed_closed", "genuine_fail"):
                kind = "fail"
                if d == "failed_closed":
                    close = "Successful"
                    gt.append(("FAILED_BUT_CLOSED", idx))
            evid.append(make_evidence(rng, rec, idx, c, kind, ts, env, force, scope))
        # distractor attachments (filename and caption agree)
        for _ in range(rng.choice([0, 1, 1, 2])):
            fname, cap = rng.choice([
                (f"change_approval_{number[-4:]}.pdf", f"CAB approval recorded for {number}, approved by {rng.choice(PERSONS)}"),
                (f"release_notes_{to_v}.docx", f"Release notes for {app} {to_v}, list of fixes included"),
                ("implementation_runbook.docx", f"Runbook steps for deployment of {app} to production nodes")])
            evid.append({"filename": fname,
                         "captured_at": (start - timedelta(days=rng.randint(1, 7))).isoformat(timespec="minutes"),
                         "text": cap, "_spans": []})
        rng.shuffle(evid)
        rec["validation_plan"] = plan
        rec["evidence"] = evid
        rec["close_code"] = close
        recs.append(rec)
        truth.append({"number": number, "findings": [{"type": t, "key": k} for t, k in gt],
                      "plan_checks": plan_checks})
    return recs, truth


def ner_rows(recs):
    """BIO rows from the exact spans known at generation time."""
    tok = re.compile(r"\w+(?:[.\-/:]\w+)*|[^\w\s]")
    rows = []
    for r in recs:
        for e in r["evidence"]:
            if not e["_spans"]:
                continue
            toks = [(m.group(0), m.start(), m.end()) for m in tok.finditer(e["text"])]
            tags = ["O"] * len(toks)
            for a, b, lab in e["_spans"]:
                first = True
                for i, (_, s, t) in enumerate(toks):
                    if s >= a and t <= b:
                        tags[i] = ("B-" if first else "I-") + lab
                        first = False
            rows.append({"tokens": [t[0] for t in toks], "ner_tags": tags, "text": e["text"],
                         "spans": e["_spans"]})
    return rows


def write_all(n=200, seed=7):
    OUT.mkdir(parents=True, exist_ok=True)
    recs, truth = generate(n, seed)
    with open(OUT / "changes.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")} |
                               {"evidence": [{k: v for k, v in e.items() if not k.startswith("_")} for e in r["evidence"]]}) + "\n")
    with open(OUT / "ground_truth.jsonl", "w") as f:
        for t in truth:
            f.write(json.dumps(t) + "\n")
    rows = ner_rows(recs)
    with open(OUT / "ner_train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return recs, truth, rows


if __name__ == "__main__":
    recs, truth, rows = write_all()
    from collections import Counter
    c = Counter(f["type"] for t in truth for f in t["findings"])
    print(len(recs), "records;", sum(1 for t in truth if t["findings"]), "with >=1 planted finding")
    print(dict(c)); print(len(rows), "NER sentences")
