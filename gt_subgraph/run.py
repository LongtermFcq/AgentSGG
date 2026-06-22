"""Entry point: per-scan loop. Phase 0 -> Phase A -> Phase B -> output + validation.

Usage:
  python run.py                      # default sample scans
  python run.py <scan_id> [<scan_id> ...]
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl
import mesh_instance as mi
import gt_builder as gb
import output as op

ROOT = "/home/data16t1/fengchangqun/AgentSGG/3RScan"
OUT = os.path.join("/home/data16t1/fengchangqun/AgentSGG", "gt_subgraph", "out")
os.makedirs(OUT, exist_ok=True)

# split lookup so we read relationships from the right file (and record split)
SPLIT = {}
for sp in ["train", "validation"]:
    f = os.path.join(ROOT, "3DSSG_subset", f"{sp}_scans.txt")
    if os.path.exists(f):
        with open(f) as fh:
            for line in fh:
                SPLIT[line.strip()] = sp

DEFAULT_SCANS = [
    "7272e16c-a01b-20f6-8961-a0927b4a7629",
    "7272e161-a01b-20f6-8b5a-0b97efeb6545",
    "f62fd5fd-9a3f-2f44-883a-1e5cf819608e",
]


def id_check(scan_id, relationships, inst_ids, labels, renderable_ids=None):
    """Phase 0 step 6: endpoint coverage + readable spot-check.

    `inst_ids` is the semseg objectId set. Renderability is reported separately
    so annotation-only objects do not get mislabeled as ID mismatches.
    """
    endpoints = {r[0] for r in relationships} | {r[1] for r in relationships}
    missing_records = []
    for r in relationships:
        missing_ids = [x for x in (r[0], r[1]) if x not in inst_ids]
        if missing_ids:
            missing_records.append({
                "scan_id": scan_id,
                "relation": list(r),
                "missing_ids": missing_ids,
            })
    missing_endpoint_ids = sorted(endpoints - inst_ids)
    print(f"  [id_check] relations={len(relationships)} "
          f"endpoints={len(endpoints)} missing_relation_rows={len(missing_records)} "
          f"missing_endpoint_ids={missing_endpoint_ids if missing_endpoint_ids else []}")
    print("  [id_check] raw relation samples:", [list(r) for r in relationships[:2]])
    for rec in missing_records[:10]:
        print(f"    MISSING scan={scan_id} relation={rec['relation']} "
              f"missing_ids={rec['missing_ids']}")
    for r in relationships[:5]:
        s, o, pid, pn = r
        print(f"    {labels.get(s,'?')}(id={s}) --{pn}--> {labels.get(o,'?')}(id={o})")

    report = {
        "relations_total": len(relationships),
        "relations_with_missing_endpoint": len(missing_records),
        "missing_endpoint_ids": missing_endpoint_ids,
        "missing_relation_samples": missing_records[:20],
    }
    if renderable_ids is not None:
        report["relation_endpoint_ids_without_renderable_geometry"] = sorted(
            int(x) for x in endpoints - set(renderable_ids) if x in inst_ids
        )
    if relationships and len(missing_records) / len(relationships) > 0.05:
        raise ValueError(
            f"[{scan_id}] too many relationship rows have missing semseg endpoints: "
            f"{len(missing_records)}/{len(relationships)}")
    return report


def run_scan(scan_id, cfg):
    scan_dir = os.path.join(ROOT, scan_id)
    split = SPLIT.get(scan_id)
    print(f"\n=== {scan_id}  (split={split}) ===")

    # Phase 0
    verts, faces = dl.load_mesh(scan_dir)
    seg_indices = dl.load_segs(scan_dir)
    seg_to_instance, labels = dl.load_semseg(scan_dir)
    ply = dl.load_ply_objectid(scan_dir, len(verts))
    inv = dl.validate_invariants(verts, faces, seg_indices, ply, scan_id)
    print(f"  [invariant] obj_v={inv['obj_vertices']} seg={inv['seg_indices']} "
          f"ply={inv['ply_vertices']} faces={inv['faces']} "
          f"(max_face_idx={inv['face_index_max']}, refs {inv['verts_referenced']}/"
          f"{inv['obj_vertices']} verts, {inv['faces_per_vert']} f/v)")
    face_inst, n_dis = mi.build_face_to_instance(faces, seg_indices, seg_to_instance)
    face_area = mi.compute_face_areas(verts, faces)
    total_area = mi.total_area_per_instance(face_inst, face_area)
    dis = mi.crosscheck_with_ply(face_inst, faces, ply)
    diag = mi.instance_diagnostics(face_inst, face_area, seg_indices,
                                   seg_to_instance, faces, labels)
    summ = mi.summarize_diagnostics(diag)
    annot_only = [(oid, r["label"], r["vertex_count"])
                  for oid, r in diag.items() if r["face_count"] == 0 and r["vertex_count"] > 0]
    inst_ids = set(labels.keys())
    renderable_ids = set(total_area.keys())
    print(f"  [phase0] semseg_inst={summ['n_instances']} has_geometry="
          f"{summ['has_geometry']} annotation_only={summ['annotation_only']} "
          f"empty={summ['empty']}; mixed-instance faces={n_dis}, "
          f"PLY crosscheck disagree={dis:.4%}")
    if annot_only:
        print(f"  [phase0] annotation-only (no mesh faces, never committable): "
              + ", ".join(f"{lbl}(id={oid},v={vc})" for oid, lbl, vc in annot_only))

    relationships, rel_file = dl.load_relationships(ROOT, scan_id, split)
    print(f"  [rel] {len(relationships)} relations from {rel_file}")
    id_report = id_check(scan_id, relationships, inst_ids, labels, renderable_ids)

    # Phase A
    K, W, H, _ = dl.load_intrinsics(scan_dir)
    frames = list(dl.iter_frames(scan_dir))
    res = gb.build_scan(verts, faces, face_inst, face_area, total_area,
                        frames, K, W, H, cfg, collect_stats=False)

    # Phase B
    edges, missing = gb.activate_edges(relationships, res["commit_time"], inst_ids)

    n_committed = sum(1 for v in res["commit_time"].values() if v is not None)
    n_active = sum(1 for e in edges if e["activation_time"] is not None)
    reasons = {}
    for v in res["commit_meta"].values():
        if v:
            reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
    print(f"  [phaseA] committed {n_committed}/{len(renderable_ids)} renderable nodes, reasons={reasons}")
    print(f"  [phaseB] active edges {n_active}/{len(edges)} (missing endpoints={missing})")
    unc = res["debug_uncommitted"]
    if unc:
        print(f"  [debug] {len(unc)} uncommitted renderable instance(s):")
        for i, d in unc.items():
            print(f"    id={i} {labels.get(i,'?')}: max_pix_vis={d['max_pix_vis']} "
                  f"max_pix_full={d['max_pix_full']} max_vis_ratio={d['max_vis_ratio']:.3f} "
                  f"max_cum={d['max_cumulative_area_ratio']:.3f} "
                  f"valid_frames={d['valid_observation_frame_count']} "
                  f"filt_pixmin={d['filtered_by_pix_min_count']} "
                  f"filt_visratio={d['filtered_by_vis_ratio_count']}")

    out = op.build_output(scan_id, res, edges, labels, cfg, id_report)
    path = os.path.join(OUT, f"gt_{scan_id[:8]}.json")
    op.save(out, path)

    # validation: commit timeline + materialize sanity (edge never before endpoints)
    committed = [(nd["commit_time"], nid, nd["label"], nd["commit_reason"])
                 for nid, nd in out["nodes"].items()]
    committed.sort()
    print("  [timeline] first commits:",
          [(t, lbl) for t, _, lbl, _ in committed[:6]])
    T = res["num_processed_frames"]
    if T > 0:
        for t in sorted(set([T // 4, T // 2, T - 1])):
            nodes_t, edges_t = op.materialize(out, t)
            bad = sum(1 for e in edges_t
                      if nodes_t.get(e["subject"], {}).get("commit_time", 1e9) > t
                      or nodes_t.get(e["object"], {}).get("commit_time", 1e9) > t)
            print(f"    materialize(t={t}): {len(nodes_t)} nodes, {len(edges_t)} edges, "
                  f"edge-before-endpoint violations={bad}")
    print(f"  saved {path}")
    return out


def main():
    scans = sys.argv[1:] or DEFAULT_SCANS
    cfg = gb.Config(TAU_INST_PIX_MIN=20, TAU_INST_VIS_RATIO=0.10, TAU_FACE_PIX=2,
                    TAU_STRONG=0.6, TAU_COMMIT=0.4,
                    ENABLE_PERSIST=True, K=3, TAU_PERSIST=0.10)
    print("config:", cfg)
    for sc in scans:
        run_scan(sc, cfg)


if __name__ == "__main__":
    main()
