"""Phase A (per-frame evidence + commit) and Phase B (edge activation).

All visibility measures use AREA ratios, never face counts. Commit is monotonic.
The per-frame loop also records raw ratios so stats.py can pick thresholds.
"""
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from renderer import FaceRenderer
from mesh_instance import BG


@dataclass
class Config:
    TAU_INST_PIX: int = 100      # instance per-frame total-pixel floor (fragment filter)
    TAU_FACE_PIX: int = 2        # per-face pixel floor (block "a corner counts as whole face")
    TAU_STRONG: float = 0.30     # single-frame visible-area ratio -> commit now
    TAU_COMMIT: float = 0.30     # cumulative visible-area ratio -> commit
    ENABLE_PERSIST: bool = False
    K: int = 3                   # frames of recurrence required for persistent commit
    TAU_PERSIST: float = 0.10    # weak-evidence floor for persistent branch


def build_scan(verts, faces, face_inst, face_area, total_area,
               frames, K_intr, W, H, cfg: Config, collect_stats=False):
    """Run Phase A over all valid frames. Returns dict with node commit info,
    t->frame map, and (optionally) raw per-frame ratio samples for stats."""
    rend = FaceRenderer(verts, faces)
    n_faces = len(faces)

    # per-instance face index lists (for area lookups)
    faces_of = {}
    for inst in total_area:
        faces_of[inst] = np.nonzero(face_inst == inst)[0]

    seen_faces = {i: np.zeros(n_faces, dtype=bool) for i in total_area}
    frame_count = {i: 0 for i in total_area}
    commit_time = {i: None for i in total_area}
    commit_meta = {i: None for i in total_area}

    t_to_frame = {}
    stats = {"single": [], "cumulative": [], "inst_pix": []} if collect_stats else None

    for t, (fid, pose) in enumerate(tqdm(frames, desc="frames", leave=False)):
        t_to_frame[t] = fid
        prim = rend.render_face_ids(K_intr, pose, W, H)  # (H,W) face id, -1 = miss
        hit = prim >= 0
        flat = prim[hit]
        if flat.size == 0:
            continue
        # pixels per face
        pix_per_face = np.bincount(flat, minlength=n_faces)
        # pixels per instance
        face_i = face_inst  # alias
        inst_of_pix = face_i[flat]
        # aggregate instance pixel counts (skip BG)
        inst_ids = inst_of_pix[inst_of_pix != BG]
        if inst_ids.size == 0:
            continue
        uniq, cnt = np.unique(inst_ids, return_counts=True)
        pix_per_inst = dict(zip(uniq.tolist(), cnt.tolist()))

        for i, ipix in pix_per_inst.items():
            if collect_stats:
                stats["inst_pix"].append(ipix)
            if ipix < cfg.TAU_INST_PIX:
                continue  # fragment

            fi = faces_of[i]
            visible = fi[pix_per_face[fi] >= cfg.TAU_FACE_PIX]
            if visible.size == 0:
                continue
            single_area = face_area[visible].sum() / total_area[i]

            seen_faces[i][visible] = True
            cum_area = face_area[seen_faces[i]].sum() / total_area[i]
            frame_count[i] += 1

            if collect_stats:
                stats["single"].append(single_area)
                stats["cumulative"].append(cum_area)

            if commit_time[i] is None:
                reason = None
                if single_area >= cfg.TAU_STRONG:
                    reason = "strong_single"
                elif cum_area >= cfg.TAU_COMMIT:
                    reason = "cumulative"
                elif (cfg.ENABLE_PERSIST and frame_count[i] >= cfg.K
                      and cum_area >= cfg.TAU_PERSIST):
                    reason = "persistent"
                if reason:
                    commit_time[i] = t
                    commit_meta[i] = {
                        "reason": reason,
                        "single_ratio_at_commit": float(single_area),
                        "cumulative_ratio_at_commit": float(cum_area),
                        "pixel_count_at_commit": int(ipix),
                        "frame_count_at_commit": int(frame_count[i]),
                    }

    return {
        "t_to_frame": t_to_frame,
        "commit_time": commit_time,
        "commit_meta": commit_meta,
        "num_processed_frames": len(frames),
        "stats": stats,
    }


def activate_edges(relationships, commit_time):
    """Phase B: edge activation = max(commit_time[subj], commit_time[obj]).
    Keeps direction + multi-edges; logs endpoints that never commit / are missing."""
    edges = []
    missing = 0
    for subj, obj, pid, pname in relationships:
        ct_i = commit_time.get(subj)
        ct_j = commit_time.get(obj)
        if subj not in commit_time or obj not in commit_time:
            missing += 1
            act = None
        elif ct_i is None or ct_j is None:
            act = None
        else:
            act = max(ct_i, ct_j)
        edges.append({
            "subject": subj, "predicate": pname, "object": obj,
            "activation_time": act,
        })
    return edges, missing
