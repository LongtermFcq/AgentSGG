"""Output contract: compact per-scan JSON + materialize(t) -> G*_{<=t}.

The JSON does NOT store a full graph per frame; nodes carry their commit_time
and edges their activation_time, and materialize(t) reconstructs the subgraph
for any t on demand (used heavily by downstream reward/oracle stages).

NODE_SCOPE controls which committed instances appear in the delivered graph:
  all_renderable  - every instance that passed visibility commit (legacy behaviour)
  rel_endpoints   - only subject/object ids from the relationship file used at build
"""
import json
from dataclasses import asdict

try:
    from gt_builder import NODE_SCOPE_ALL_RENDERABLE, NODE_SCOPE_REL_ENDPOINTS
except ImportError:
    NODE_SCOPE_ALL_RENDERABLE = "all_renderable"
    NODE_SCOPE_REL_ENDPOINTS = "rel_endpoints"

NODE_SCOPES = (NODE_SCOPE_ALL_RENDERABLE, NODE_SCOPE_REL_ENDPOINTS)


def relationship_endpoint_ids(relationships):
    """Unique instance ids that appear as subject or object in relationship rows."""
    ids = set()
    for subj, obj, _pid, _pname in relationships:
        ids.add(int(subj))
        ids.add(int(obj))
    return sorted(ids)


def _resolve_node_scope(out, node_scope=None):
    if node_scope is not None:
        if node_scope not in NODE_SCOPES:
            raise ValueError(f"node_scope must be one of {NODE_SCOPES}, got {node_scope!r}")
        return node_scope
    return out.get("config", {}).get("NODE_SCOPE", NODE_SCOPE_ALL_RENDERABLE)


def _allowed_node_ids(out, node_scope=None):
    scope = _resolve_node_scope(out, node_scope)
    if scope == NODE_SCOPE_REL_ENDPOINTS:
        return {str(i) for i in out.get("rel_endpoint_ids", [])}
    return None


def _all_committed_nodes(out):
    """Every committed node record, including build-time NODE_SCOPE exclusions."""
    pool = dict(out["nodes"])
    for nid, rec in out.get("debug", {}).get("nodes_excluded_by_scope", {}).items():
        pool.setdefault(nid, rec)
    return pool


def _node_record(i, ct, meta, labels):
    return {
        "commit_time": ct,
        "label": labels.get(i, "?"),
        "commit_reason": meta["reason"],
        "single_ratio_at_commit": round(meta["single_ratio_at_commit"], 4),
        "cumulative_ratio_at_commit": round(meta["cumulative_ratio_at_commit"], 4),
        "pixel_count_at_commit": meta["pixel_count_at_commit"],
        "frame_count_at_commit": meta["frame_count_at_commit"],
    }


def build_output(scan_id, res, edges, labels, cfg, id_report, rel_endpoint_ids=None):
    rel_endpoint_ids = rel_endpoint_ids or []
    allowed = (
        {str(i) for i in rel_endpoint_ids}
        if cfg.NODE_SCOPE == NODE_SCOPE_REL_ENDPOINTS
        else None
    )

    nodes = {}
    excluded = {}
    for i, ct in res["commit_time"].items():
        if ct is None:
            continue
        meta = res["commit_meta"][i]
        rec = _node_record(i, ct, meta, labels)
        nid = str(i)
        if allowed is not None and nid not in allowed:
            excluded[nid] = rec
        else:
            nodes[nid] = rec

    out = {
        "scan_id": scan_id,
        "num_processed_frames": res["num_processed_frames"],
        "t_to_frame": {str(t): f for t, f in res["t_to_frame"].items()},
        "config": asdict(cfg),
        "rel_endpoint_ids": [int(i) for i in rel_endpoint_ids],
        "nodes": nodes,
        "edges": [
            {"subject": str(e["subject"]), "predicate": e["predicate"],
             "object": str(e["object"]), "activation_time": e["activation_time"]}
            for e in edges
        ],
        "id_check": id_report,
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
            },
            "nodes_excluded_by_scope": excluded,
        },
    }
    return out


def materialize(out, t, node_scope=None):
    """Return (nodes, edges) of G*_{<=t}: nodes committed by t, directed edges
    activated by t. Edges are kept directional and multi-edges preserved.

    node_scope overrides out['config']['NODE_SCOPE'] when provided. When the JSON
    was built with a narrower scope, widening via node_scope='all_renderable'
    restores nodes from debug.nodes_excluded_by_scope.
    """
    allowed = _allowed_node_ids(out, node_scope)
    nodes = {
        nid: nd for nid, nd in _all_committed_nodes(out).items()
        if nd["commit_time"] is not None and nd["commit_time"] <= t
        and (allowed is None or nid in allowed)
    }
    edges = [
        e for e in out["edges"]
        if e["activation_time"] is not None and e["activation_time"] <= t
        and (allowed is None or (e["subject"] in allowed and e["object"] in allowed))
    ]
    return nodes, edges


def save(out, path):
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
