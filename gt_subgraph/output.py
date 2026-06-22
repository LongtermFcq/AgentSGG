"""Output contract: compact per-scan JSON + materialize(t) -> G*_{<=t}.

The JSON does NOT store a full graph per frame; nodes carry their commit_time
and edges their activation_time, and materialize(t) reconstructs the subgraph
for any t on demand (used heavily by downstream reward/oracle stages)."""
import json
from dataclasses import asdict


def build_output(scan_id, res, edges, labels, cfg, relations_total, missing):
    nodes = {}
    for i, ct in res["commit_time"].items():
        if ct is None:
            continue  # only committed instances become nodes
        meta = res["commit_meta"][i]
        nodes[str(i)] = {
            "commit_time": ct,
            "label": labels.get(i, "?"),
            "commit_reason": meta["reason"],
            "single_ratio_at_commit": round(meta["single_ratio_at_commit"], 4),
            "cumulative_ratio_at_commit": round(meta["cumulative_ratio_at_commit"], 4),
            "pixel_count_at_commit": meta["pixel_count_at_commit"],
            "frame_count_at_commit": meta["frame_count_at_commit"],
        }
    out = {
        "scan_id": scan_id,
        "num_processed_frames": res["num_processed_frames"],
        "t_to_frame": {str(t): f for t, f in res["t_to_frame"].items()},
        "config": asdict(cfg),
        "nodes": nodes,
        "edges": [
            {"subject": str(e["subject"]), "predicate": e["predicate"],
             "object": str(e["object"]), "activation_time": e["activation_time"]}
            for e in edges
        ],
        "id_check": {
            "relations_total": relations_total,
            "relations_with_missing_endpoint": missing,
        },
        "debug": {
            "uncommitted_renderable": {
                str(i): {
                    "label": labels.get(int(i), "?"),
                    "max_pix_vis": d["max_pix_vis"],
                    "max_pix_full": d["max_pix_full"],
                    "max_vis_ratio": round(d["max_vis_ratio"], 4),
                    "max_single_area_ratio": round(d["max_single_area_ratio"], 4),
                    "max_cumulative_area_ratio": round(d["max_cumulative_area_ratio"], 4),
                    "valid_observation_frame_count": d["valid_observation_frame_count"],
                    "filtered_by_pix_min_count": d["filtered_by_pix_min_count"],
                    "filtered_by_vis_ratio_count": d["filtered_by_vis_ratio_count"],
                }
                for i, d in res["debug_uncommitted"].items()
            }
        },
    }
    return out


def materialize(out, t):
    """Return (nodes, edges) of G*_{<=t}: nodes committed by t, directed edges
    activated by t. Edges are kept directional and multi-edges preserved."""
    nodes = {nid: nd for nid, nd in out["nodes"].items()
             if nd["commit_time"] is not None and nd["commit_time"] <= t}
    edges = [e for e in out["edges"]
             if e["activation_time"] is not None and e["activation_time"] <= t]
    return nodes, edges


def save(out, path):
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
