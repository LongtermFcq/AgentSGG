"""Compare original 3DSSG annotation vs final temporal GT subgraph (side-by-side).

Uses the same top-down floor-plan style as visualize.py so you can eyeball whether
materialize(out, T-1) matches the source annotation.

Usage:
  python visualize_compare.py                          # default sample scan
  python visualize_compare.py <scan_id>
  python visualize_compare.py <scan_id> --rel full     # also show full relationships.json
  python visualize_compare.py <scan_id> --rel split    # only split file (default)
  python visualize_compare.py <scan_id> --rel both     # split + full + temporal
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl
import mesh_instance as mi
import output as op
import visualize as viz
from paths import ROOT, DEMO_VIZ, DEFAULT_SCANS, gt_json_path, ensure_demo_dirs

ensure_demo_dirs()

SPLIT = {}
for sp in ("train", "validation"):
    p = os.path.join(ROOT, "3DSSG_subset", f"{sp}_scans.txt")
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                sid = line.strip().rstrip(",")
                if sid:
                    SPLIT[sid] = sp


def _edge_key(e):
    return (str(e["subject"]), e["predicate"], str(e["object"]))


def build_annotation_out(scan_id, rel_mode="split"):
    """Build a pseudo GT JSON from raw 3DSSG relationships + semseg labels.

    rel_mode:
      split  - same file run.py uses (relationships_{train,validation}.json)
      full   - relationships.json for this scan (complete per-scan graph)
    """
    scan_dir = os.path.join(ROOT, scan_id)
    split = SPLIT.get(scan_id) if rel_mode == "split" else None
    relationships, rel_file = dl.load_relationships(ROOT, scan_id, split)
    if rel_mode == "full":
        fp = os.path.join(ROOT, "3DSSG_subset", "relationships.json")
        if os.path.exists(fp):
            data = json.load(open(fp))
            for s in data["scans"]:
                if s["scan"] == scan_id:
                    relationships = [tuple(r) for r in s["relationships"]]
                    rel_file = "relationships.json"
                    break

    seg_to_instance, labels = dl.load_semseg(scan_dir)
    verts, faces = dl.load_mesh(scan_dir)
    seg = dl.load_segs(scan_dir)
    face_inst, _ = mi.build_face_to_instance(faces, seg, seg_to_instance)
    face_area = mi.compute_face_areas(verts, faces)
    renderable = set(mi.total_area_per_instance(face_inst, face_area))

    endpoints = set()
    for subj, obj, _pid, _pname in relationships:
        endpoints.add(int(subj))
        endpoints.add(int(obj))

    nodes = {}
    for oid in sorted(endpoints):
        nodes[str(oid)] = {
            "commit_time": 0,
            "label": labels.get(oid, "?"),
            "commit_reason": "annotation",
            "renderable": oid in renderable,
        }

    edges = [
        {
            "subject": str(subj),
            "predicate": pname,
            "object": str(obj),
            "activation_time": 0,
        }
        for subj, obj, _pid, pname in relationships
    ]

    return {
        "scan_id": scan_id,
        "num_processed_frames": 1,
        "nodes": nodes,
        "edges": edges,
        "_source": rel_file,
        "_rel_mode": rel_mode,
    }


def diff_report(annot_out, temporal_out, t_final):
    """Compare annotation vs materialized temporal graph at t_final."""
    t_nodes, t_edges = op.materialize(temporal_out, t_final)
    a_nodes = annot_out["nodes"]
    a_edges = [e for e in annot_out["edges"] if e["activation_time"] is not None]

    a_nids = set(a_nodes)
    t_nids = set(t_nodes)
    a_ekeys = {_edge_key(e) for e in a_edges}
    t_ekeys = {_edge_key(e) for e in t_edges}

    def node_detail(nids, src):
        return {
            nid: {
                "label": src[nid].get("label", "?"),
                "renderable": src[nid].get("renderable", True),
            }
            for nid in sorted(nids, key=lambda x: int(x))
        }

    return {
        "annotation_source": annot_out.get("_source"),
        "t_final": t_final,
        "nodes": {
            "annotation": len(a_nids),
            "temporal": len(t_nids),
            "only_in_annotation": node_detail(a_nids - t_nids, a_nodes),
            "only_in_temporal": node_detail(t_nids - a_nids, t_nodes),
            "in_both": sorted(a_nids & t_nids, key=lambda x: int(x)),
        },
        "edges": {
            "annotation": len(a_ekeys),
            "temporal": len(t_ekeys),
            "only_in_annotation": sorted(a_ekeys - t_ekeys),
            "only_in_temporal": sorted(t_ekeys - a_ekeys),
            "in_both": sorted(a_ekeys & t_ekeys),
        },
    }


def compare_panel(panels, cents, up, labels, path, suptitle):
    """panels: list of (title, out_dict, t)"""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 7))
    if n == 1:
        axes = [axes]

    all_preds = []
    for _, out, _ in panels:
        all_preds.extend(e["predicate"] for e in out["edges"])
    pred_palette = viz._pred_palette(all_preds)

    for ax, (title, out, t) in zip(axes, panels):
        nodes, edges = op.materialize(out, t)
        viz.draw_graph(ax, out, t, cents, up, labels, pred_palette,
                       title=f"{title}\n({len(nodes)} nodes, {len(edges)} edges)")

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(scan_id, rel_mode="split"):
    gt_path = gt_json_path(scan_id)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"GT JSON not found: {gt_path}\nRun run.py for this scan first.")

    with open(gt_path) as f:
        temporal = json.load(f)

    scan_dir = os.path.join(ROOT, scan_id)
    cents, up, labels = viz.load_centroids(scan_dir)
    t_final = max(0, temporal["num_processed_frames"] - 1)

    panels = []
    reports = []

    if rel_mode in ("split", "both"):
        annot_split = build_annotation_out(scan_id, rel_mode="split")
        panels.append((
            f"Annotation (split)\n{annot_split['_source']}",
            annot_split,
            0,
        ))
        reports.append(("split", diff_report(annot_split, temporal, t_final)))

    if rel_mode in ("full", "both"):
        annot_full = build_annotation_out(scan_id, rel_mode="full")
        panels.append((
            f"Annotation (full)\n{annot_full['_source']}",
            annot_full,
            0,
        ))
        if rel_mode == "full":
            reports.append(("full", diff_report(annot_full, temporal, t_final)))

    panels.append((
        f"Temporal G* at t={t_final}\n(final frame)",
        temporal,
        t_final,
    ))

    if rel_mode == "both":
        reports.append(("full", diff_report(annot_full, temporal, t_final)))

    tag = scan_id[:8]
    img_path = os.path.join(DEMO_VIZ, f"compare_{tag}_{rel_mode}.png")
    compare_panel(
        panels,
        cents, up, labels,
        img_path,
        suptitle=f"{scan_id}  annotation vs temporal GT (top-down)",
    )
    print(f"  saved {img_path}")

    report_path = os.path.join(DEMO_VIZ, f"compare_{tag}_{rel_mode}.json")
    out_report = {
        "scan_id": scan_id,
        "rel_mode": rel_mode,
        "t_final": t_final,
        "comparisons": {name: rep for name, rep in reports if rep},
    }
    with open(report_path, "w") as f:
        json.dump(out_report, f, indent=2, ensure_ascii=False)
    print(f"  saved {report_path}")

    for name, rep in reports:
        if not rep:
            continue
        print(f"\n  --- diff ({name}) ---")
        print(f"  nodes: annot={rep['nodes']['annotation']} temporal={rep['nodes']['temporal']} "
              f"both={len(rep['nodes']['in_both'])}")
        if rep["nodes"]["only_in_annotation"]:
            print(f"  only in annotation: {rep['nodes']['only_in_annotation']}")
        if rep["nodes"]["only_in_temporal"]:
            print(f"  only in temporal: {rep['nodes']['only_in_temporal']}")
        print(f"  edges: annot={rep['edges']['annotation']} temporal={rep['edges']['temporal']} "
              f"both={len(rep['edges']['in_both'])}")
        if rep["edges"]["only_in_annotation"]:
            print(f"  edges only in annotation ({len(rep['edges']['only_in_annotation'])}):")
            for e in rep["edges"]["only_in_annotation"][:8]:
                print(f"    {e[0]} --{e[1]}--> {e[2]}")
            if len(rep["edges"]["only_in_annotation"]) > 8:
                print(f"    ... +{len(rep['edges']['only_in_annotation']) - 8} more")
        if rep["edges"]["only_in_temporal"]:
            print(f"  edges only in temporal: {rep['edges']['only_in_temporal']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare annotation vs temporal GT subgraph")
    parser.add_argument("scans", nargs="*", default=[DEFAULT_SCANS[0]])
    parser.add_argument(
        "--rel", choices=("split", "full", "both"), default="split",
        help="annotation source: split file (default), full relationships.json, or both",
    )
    args = parser.parse_args()
    for sc in args.scans:
        print(f"\n=== compare {sc} (rel={args.rel}) ===")
        run(sc, rel_mode=args.rel)
