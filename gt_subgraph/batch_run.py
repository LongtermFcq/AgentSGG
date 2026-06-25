#!/usr/bin/env python3
"""Overnight batch: build downstream-usable temporal GT JSON for FRACTION of the
processable scans, resumable, with ntfy push at every 10% milestone.

Design notes:
  - "processable" = scans that have semseg.v2.json on disk AND appear in the
    full 3DSSG_subset/relationships.json (train + val pool, 1335 scans). Test
    scans have no relations and are excluded.
  - All eligible scans are pooled together, shuffled with a FIXED seed, and the
    first FRACTION are this run's target. The full ordering is persisted
    to outputs/batch_manifest.json so the remainder can be run later with
    the same deterministic split, and so a restart resumes the same set.
  - Only the GT JSON (outputs/gt/gt_*.json) is produced in the main loop --
    nothing else. Scans whose JSON already exists are skipped (resume).
  - Per-scan failures are caught + logged (outputs/batch_failures.log) and do
    NOT abort the batch.
  - After the build loop completes, a few random DONE scans get a comparison
    image (visualize_compare --rel full) and an animated gif (visualize.py).
    No textual report, no panel/timeline pngs.
"""
import os
import sys
import json
import time
import random
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gt_builder as gb
import run as runmod
import visualize_compare as vc
import visualize as viz
from paths import (ROOT, GT_SUBGRAPH_DIR, DEMO_VIZ, gt_json_path,
                   ensure_gt_out, ensure_demo_dirs)

NTFY_TOPIC = "https://ntfy.sh/FCQ"
SEED = 20260624
FRACTION = 1.0
N_POSTVIZ = 4  # how many random scans get compare-png + gif at the end

OUT_DIR = os.path.join(GT_SUBGRAPH_DIR, "outputs")
MANIFEST = os.path.join(OUT_DIR, "batch_manifest.json")
FAIL_LOG = os.path.join(OUT_DIR, "batch_failures.log")


def push(msg, title="GT batch", priority=None):
    """Best-effort ntfy push; never raises."""
    try:
        cmd = ["curl", "-s", "--max-time", "15", "-H", f"Title: {title}"]
        if priority:
            cmd += ["-H", f"Priority: {priority}"]
        cmd += ["-d", msg, NTFY_TOPIC]
        subprocess.run(cmd, check=False, capture_output=True)
    except Exception as exc:
        print(f"[push] failed: {exc}", flush=True)


def eligible_scans():
    disk = {d for d in os.listdir(ROOT)
            if len(d) == 36 and d[8] == "-"
            and os.path.isdir(os.path.join(ROOT, d))
            and os.path.exists(os.path.join(ROOT, d, "semseg.v2.json"))}
    rel = json.load(open(os.path.join(ROOT, "3DSSG_subset", "relationships.json")))
    relids = {s["scan"] for s in rel["scans"]}
    return sorted(disk & relids)


def select_target():
    elig = eligible_scans()
    rng = random.Random(SEED)
    order = elig[:]
    rng.shuffle(order)
    n_target = int(len(order) * FRACTION)
    selected = sorted(order[:n_target])
    rest = sorted(order[n_target:])
    json.dump({"seed": SEED, "fraction": FRACTION, "n_eligible": len(elig),
               "n_target": n_target, "selected": selected, "rest": rest},
              open(MANIFEST, "w"), indent=1)
    return elig, selected


def make_gif(sc):
    """Animated top-down GT graph growing over t -- gif only (no panel/timeline)."""
    out = json.load(open(gt_json_path(sc)))
    cents, up, labels = viz.load_centroids(os.path.join(ROOT, sc))
    pal = viz._pred_palette([e["predicate"] for e in out["edges"]])
    gif = os.path.join(DEMO_VIZ, f"viz_{sc[:8]}_floorplan.gif")
    viz.animate_gif(out, cents, up, labels, pal, gif)
    return gif


def main():
    ensure_gt_out()
    elig, selected = select_target()
    n = len(selected)
    cfg = gb.Config(
        TAU_INST_PIX_MIN=20, TAU_INST_VIS_RATIO=0.10, TAU_FACE_PIX=2,
        TAU_STRONG=0.6, TAU_COMMIT=0.4,
        ENABLE_PERSIST=True, K=3, TAU_PERSIST=0.10,
        NODE_SCOPE=gb.NODE_SCOPE_REL_ENDPOINTS,
    )
    print(f"[batch] eligible={len(elig)} target({FRACTION*100:.0f}%)={n}  cfg={cfg}", flush=True)
    push(f"start: target {n}/{len(elig)} scans ({FRACTION*100:.0f}%), only GT JSON")

    t0 = time.time()
    ok = skipped = fail = 0
    last_ms = 0
    for i, sc in enumerate(selected, 1):
        if os.path.exists(gt_json_path(sc)):
            skipped += 1
            ok += 1
        else:
            try:
                runmod.run_scan(sc, cfg)
                ok += 1
            except Exception as exc:
                fail += 1
                import traceback
                with open(FAIL_LOG, "a") as fh:
                    fh.write(f"=== {sc} ===\n{traceback.format_exc()}\n")
                print(f"[batch] FAIL {sc}: {exc}", flush=True)

        ms = i * 10 // n
        if ms > last_ms and ms < 10:
            last_ms = ms
            el = (time.time() - t0) / 60.0
            rate = (time.time() - t0) / max(i, 1)
            eta = rate * (n - i) / 60.0
            push(f"{ms*10}%  {i}/{n} done (ok={ok} skip={skipped} fail={fail}) "
                 f"elapsed={el:.0f}min eta={eta:.0f}min")

    el = (time.time() - t0) / 60.0
    print(f"[batch] BUILD DONE ok={ok} skipped={skipped} fail={fail} {el:.0f}min", flush=True)
    push(f"BUILD DONE: {ok}/{n} ok ({skipped} resumed), {fail} fail, {el:.0f}min",
         priority="high")

    # --- post: a few random DONE scans -> compare png + gif only ---
    ensure_demo_dirs()
    done_scans = [s for s in selected if os.path.exists(gt_json_path(s))]
    rng2 = random.Random(SEED + 1)
    sample = rng2.sample(done_scans, min(N_POSTVIZ, len(done_scans)))
    print(f"[batch] post-viz sample: {sample}", flush=True)
    for sc in sample:
        try:
            vc.run(sc, rel_mode="full")           # comparison image (+ its json)
        except Exception as exc:
            print(f"[batch] compare FAIL {sc}: {exc}", flush=True)
        try:
            make_gif(sc)                            # animated gif only
        except Exception as exc:
            print(f"[batch] gif FAIL {sc}: {exc}", flush=True)
    push(f"post-viz done (compare+gif) for {len(sample)} scans: "
         f"{', '.join(s[:8] for s in sample)}")
    print("[batch] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
