"""Evaluate the audit copilot against the planted ground truth.

  python -m change_audit.evaluate                      # tfidf backend, calibrate on dev, report on test
  python -m change_audit.evaluate --backend minilm     # same, with the Part III MiniLM embedder (needs HF access)
"""
from __future__ import annotations
import argparse, json, time, itertools
from collections import defaultdict
from pathlib import Path

from .audit import Auditor, to_pred_keys
from .extract import extract_entities
from .standard import Standard
from . import synth

DATA = Path(__file__).resolve().parent.parent / "data" / "change_audit"
OUT = Path(__file__).resolve().parent.parent / "outputs"
N_DEV = 60


def load():
    recs = [json.loads(l) for l in open(DATA / "changes.jsonl")]
    truth = {t["number"]: t for t in map(json.loads, open(DATA / "ground_truth.jsonl"))}
    return recs, truth


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(auditor, recs, truth, collect=False):
    per = defaultdict(lambda: [0, 0, 0])
    rec_ok = 0
    errors = []
    t0 = time.perf_counter()
    for r in recs:
        pred = to_pred_keys(auditor.audit(r))
        gold = {(f["type"], f["key"]) for f in truth[r["number"]]["findings"]}
        for t, k in pred & gold:
            per[t][0] += 1
        for t, k in pred - gold:
            per[t][1] += 1
            if collect: errors.append(("FP", r["number"], t, k))
        for t, k in gold - pred:
            per[t][2] += 1
            if collect: errors.append(("FN", r["number"], t, k))
        rec_ok += (bool(pred) == bool(gold))
    dt = time.perf_counter() - t0
    tot = [sum(v[i] for v in per.values()) for i in range(3)]
    out = {"per_type": {t: dict(zip(("tp", "fp", "fn"), v)) | dict(zip(("precision", "recall", "f1"), prf(*v)))
                        for t, v in sorted(per.items())},
           "overall": dict(zip(("tp", "fp", "fn"), tot)) | dict(zip(("precision", "recall", "f1"), prf(*tot))),
           "ticket_flag_accuracy": rec_ok / len(recs), "n_tickets": len(recs),
           "sec_per_ticket": dt / len(recs)}
    return (out, errors) if collect else out


def calibrate(backend, dev, truth, fit_texts):
    best = (-1, None)
    grid_tau = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] if backend == "minilm" else \
               [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    grid_type = [0.05, 0.10, 0.15, 0.20] if backend == "tfidf" else [0.15, 0.25, 0.35]
    rows = []
    for tau, tt in itertools.product(grid_tau, grid_type):
        a = Auditor(backend, tau=tau, tau_type=tt, fit_texts=fit_texts)
        f1 = score(a, dev, truth)["overall"]["f1"]
        rows.append((tau, tt, f1))
        if f1 > best[0]:
            best = (f1, (tau, tt))
    return best[1], best[0], rows


def ner_eval(backend="regex", iou_thr=0.5):
    rows = [json.loads(l) for l in open(DATA / "ner_train.jsonl")]
    split = int(len(rows) * 0.7)
    test = rows[split:]
    per = defaultdict(lambda: [0, 0, 0])
    for r in test:
        gold = [tuple(s) for s in r["spans"]]
        pred = [(e["start"], e["end"], e["label"]) for e in extract_entities(r["text"], backend)]
        used = set()
        for a, b, lab in pred:
            hit = None
            for gi, (ga, gb, gl) in enumerate(gold):
                if gl != lab or gi in used:
                    continue
                inter = max(0, min(b, gb) - max(a, ga))
                union = max(b, gb) - min(a, ga)
                if union and inter / union >= iou_thr:
                    hit = gi
                    break
            if hit is None:
                per[lab][1] += 1
            else:
                per[lab][0] += 1
                used.add(hit)
        for gi, (ga, gb, gl) in enumerate(gold):
            if gi not in used:
                per[gl][2] += 1
    res = {l: dict(zip(("tp", "fp", "fn"), v)) | dict(zip(("precision", "recall", "f1"), prf(*v)))
           for l, v in sorted(per.items())}
    tot = [sum(v[i] for v in per.values()) for i in range(3)]
    res["ALL"] = dict(zip(("tp", "fp", "fn"), tot)) | dict(zip(("precision", "recall", "f1"), prf(*tot)))
    return res, len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tfidf")
    ap.add_argument("--holdout-seed", type=int, default=11, help="fresh synthetic set, never used for calibration or error analysis")
    args = ap.parse_args()
    recs, truth = load()
    dev, test = recs[:N_DEV], recs[N_DEV:]
    fit = [f"{e['filename']} {e['text']}" for r in dev for e in r["evidence"]] + [s for r in dev for s in r["validation_plan"]]
    (tau, tt), devf1, grid = calibrate(args.backend, dev, truth, fit)
    ho_recs, ho_truth_l = synth.generate(200, args.holdout_seed)
    ho_truth = {t["number"]: t for t in ho_truth_l}

    arms = {}
    for name, kw in [("full", {}), ("matching_only", {"type_aware": False})]:
        a = Auditor(args.backend, tau=tau, tau_type=tt, fit_texts=fit, **kw)
        arms[name] = {"dev_seed7_test_split": score(a, test, truth), "holdout": score(a, ho_recs, ho_truth, collect=True)}
    a = Auditor(args.backend, tau=tau, tau_type=tt, fit_texts=fit)
    ho_res, ho_err = arms["full"]["holdout"]
    arms["full"]["holdout"] = ho_res
    arms["matching_only"]["holdout"] = arms["matching_only"]["holdout"][0]
    ner, n_ner = ner_eval("regex")
    out = {"backend": args.backend, "tau": tau, "tau_type": tt, "dev_f1": devf1, "n_dev": len(dev),
           "arms": arms, "ner_regex": ner, "ner_test_sentences": n_ner, "holdout_seed": args.holdout_seed,
           "holdout_n_gold": sum(len(t["findings"]) for t in ho_truth_l),
           "holdout_errors": ho_err[:60]}
    OUT.mkdir(exist_ok=True)
    (OUT / f"metrics_{args.backend}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"backend={args.backend} tau={tau} tau_type={tt} dev_f1={devf1:.3f} (calibrated on first {len(dev)} tickets)")
    for arm in arms:
        for split in ("dev_seed7_test_split", "holdout"):
            res = arms[arm][split]
            o = res["overall"]
            print(f"\n[{arm} | {split}] tickets={res['n_tickets']} flag_acc={res['ticket_flag_accuracy']:.3f} "
                  f"s/ticket={res['sec_per_ticket']:.4f}  OVERALL P={o['precision']:.2f} R={o['recall']:.2f} F1={o['f1']:.2f} "
                  f"(tp={o['tp']} fp={o['fp']} fn={o['fn']})")
            if arm == "full":
                for t, m in res["per_type"].items():
                    print(f"    {t:18s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}  tp={m['tp']} fp={m['fp']} fn={m['fn']}")
    print("\nNER (regex pre-labeller) on", n_ner, "held-out sentences, IoU>=0.5")
    for l, m in ner.items():
        print(f"  {l:8s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")


if __name__ == "__main__":
    main()
