"""Phase A (per-frame evidence + commit) and Phase B (edge activation).

Three-tier fragment filter (replaces the old single absolute TAU_INST_PIX):
  tier1  TAU_INST_PIX_MIN : instance-level absolute pixel floor, VERY low --
                            only rejects 1..tens-of-pixels sampling noise.
  tier2  TAU_INST_VIS_RATIO: pix_vis / pix_full (occlusion-normalized 2D visible
                            ratio). pix_full comes from rendering ONLY this
                            instance's faces under the SAME camera (same K /
                            extrinsic / W / H / raycasting rules) -- no bbox or
                            2D-area approximation. Rejects sliver observations
                            (object should occupy many pixels but only a corner
                            shows). Does NOT penalise complete-but-small
                            projections (grazing angle, unoccluded -> ratio~1);
                            that reliability call is tier1's job.
  tier3  TAU_FACE_PIX     : per-face pixel floor, blocks "one corner of a big
                            face counts as the whole face area".

All visibility *evidence* measures use AREA ratios, never face counts. Commit is
monotonic. Per-instance debug extremes + filter counts are tracked so uncommitted
renderable instances can be diagnosed (never projected / only corner / too few
pixels / area evidence insufficient).
"""
from dataclasses import dataclass
import warnings
import numpy as np
from tqdm import tqdm

try:
    from .renderer import FaceRenderer, InstanceRenderer
    from .mesh_instance import BG
except ImportError:  # script execution from gt_subgraph/
    from renderer import FaceRenderer, InstanceRenderer
    from mesh_instance import BG


NODE_SCOPE_ALL_RENDERABLE = "all_renderable"
NODE_SCOPE_REL_ENDPOINTS = "rel_endpoints"


@dataclass
class Config:
    TAU_INST_PIX_MIN: int = 20       # tier1: very low absolute pixel floor (noise only)
    TAU_INST_VIS_RATIO: float = 0.10  # tier2: occlusion-normalized visible ratio
    TAU_FACE_PIX: int = 2            # tier3: per-face pixel floor
    TAU_STRONG: float = 0.60         # single-frame visible-area ratio -> commit now
    TAU_COMMIT: float = 0.40         # cumulative visible-area ratio -> commit
    ENABLE_PERSIST: bool = True      # persistent branch on (rescues weak-evidence small objects)
    K: int = 3                       # frames of recurrence required for persistent commit
    TAU_PERSIST: float = 0.10        # weak-evidence floor for persistent branch
    # Downstream node set: all_renderable = every committed instance;
    # rel_endpoints = only instances appearing in the loaded relationship file.
    NODE_SCOPE: str = NODE_SCOPE_REL_ENDPOINTS


def _init_debug():
    return {
        "max_pix_vis": 0, "max_pix_full": 0, "max_vis_ratio": 0.0,
        "max_single_area_ratio": 0.0, "max_cumulative_area_ratio": 0.0,
        "valid_observation_frame_count": 0,
        "filtered_by_pix_min_count": 0, "filtered_by_vis_ratio_count": 0,
    }


def build_scan(verts, faces, face_inst, face_area, total_area,
               frames, K_intr, W, H, cfg: Config, collect_stats=False):
    """Run Phase A over all valid frames. Returns dict with node commit info,
    t->frame map, per-instance debug for uncommitted renderable instances, and
    (optionally) raw per-frame ratio samples for stats."""
    rend = FaceRenderer(verts, faces)
    irend = InstanceRenderer(verts, faces, face_inst)
    n_faces = len(faces)

    faces_of = {inst: np.nonzero(face_inst == inst)[0] for inst in total_area}
    seen_faces = {i: np.zeros(n_faces, dtype=bool) for i in total_area}
    frame_count = {i: 0 for i in total_area}
    commit_time = {i: None for i in total_area}
    commit_meta = {i: None for i in total_area}
    debug = {i: _init_debug() for i in total_area}

    t_to_frame = {}
    stats = {"single": [], "cumulative": [], "inst_pix": [], "vis_ratio": []} \
        if collect_stats else None

    processed = 0
    for fid, pose in tqdm(frames, desc="frames", leave=False):
        try:
            prim = rend.render_face_ids(K_intr, pose, W, H)  # (H,W) face id, -1 = miss
        except Exception as exc:
            warnings.warn(f"skipping frame {fid}: render failed: {exc}", stacklevel=2)
            continue
        t = processed
        processed += 1
        t_to_frame[t] = fid
        hit = prim >= 0
        flat = prim[hit]
        if flat.size == 0:
            continue
        pix_per_face = np.bincount(flat, minlength=n_faces)
        inst_of_pix = face_inst[flat]
        inst_ids = inst_of_pix[inst_of_pix != BG]
        if inst_ids.size == 0:
            continue
        uniq, cnt = np.unique(inst_ids, return_counts=True)
        pix_per_inst = dict(zip(uniq.tolist(), cnt.tolist()))

        for i, ipix in pix_per_inst.items():
            d = debug[i]
            if d["max_pix_vis"] < ipix:
                d["max_pix_vis"] = ipix
            if collect_stats:
                stats["inst_pix"].append(ipix)

            # already committed -> no need to keep updating (monotonic; saves cost)
            if commit_time[i] is not None:
                continue

            # ---- tier1: very low absolute pixel floor (noise only) ----
            if ipix < cfg.TAU_INST_PIX_MIN:
                d["filtered_by_pix_min_count"] += 1
                continue

            # ---- tier2: occlusion-normalized visible ratio ----
            pfull = irend.pix_full(i, K_intr, pose, W, H)
            if d["max_pix_full"] < pfull:
                d["max_pix_full"] = pfull
            vis_ratio = (ipix / pfull) if pfull > 0 else 0.0
            if d["max_vis_ratio"] < vis_ratio:
                d["max_vis_ratio"] = vis_ratio
            if collect_stats and pfull > 0:
                stats["vis_ratio"].append(vis_ratio)
            if pfull == 0 or vis_ratio < cfg.TAU_INST_VIS_RATIO:
                d["filtered_by_vis_ratio_count"] += 1
                continue

            # ---- tier3: per-face pixel floor ----
            fi = faces_of[i]
            visible = fi[pix_per_face[fi] >= cfg.TAU_FACE_PIX]
            if visible.size == 0:
                continue

            single_area = float(face_area[visible].sum() / total_area[i])
            seen_faces[i][visible] = True
            cum_area = float(face_area[seen_faces[i]].sum() / total_area[i])
            frame_count[i] += 1
            d["valid_observation_frame_count"] = frame_count[i]
            if d["max_single_area_ratio"] < single_area:
                d["max_single_area_ratio"] = single_area
            if d["max_cumulative_area_ratio"] < cum_area:
                d["max_cumulative_area_ratio"] = cum_area

            if collect_stats:
                stats["single"].append(single_area)
                stats["cumulative"].append(cum_area)

            # ---- commit (monotonic, never revoked) ----
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
                        "single_ratio_at_commit": single_area,
                        "cumulative_ratio_at_commit": cum_area,
                        "pixel_count_at_commit": int(ipix),
                        "frame_count_at_commit": int(frame_count[i]),
                    }

    # debug only for renderable instances that never committed
    uncommitted = {i: d for i, d in debug.items() if commit_time[i] is None}

    return {
        "t_to_frame": t_to_frame,
        "commit_time": commit_time,
        "commit_meta": commit_meta,
        "num_processed_frames": processed,
        "debug_uncommitted": uncommitted,
        "stats": stats,
    }


def activate_edges(relationships, commit_time, known_instance_ids=None):
    """Phase B: edge activation = max(commit_time[subj], commit_time[obj]).
    Keeps direction + multi-edges. `known_instance_ids` should be the semseg
    instance-id set, so annotation-only objects are treated as known but
    uncommitted instead of being reported as ID mismatches.
    """
    known = set(commit_time) if known_instance_ids is None else set(known_instance_ids)
    edges = []
    missing = 0
    for subj, obj, pid, pname in relationships:
        subj_known = subj in known
        obj_known = obj in known
        ct_i = commit_time.get(subj)
        ct_j = commit_time.get(obj)
        if not subj_known or not obj_known:
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
