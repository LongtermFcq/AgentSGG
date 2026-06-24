"""Visualize the GT scene-graph sequence {G*_{<=t}} (handoff doc Section 4.1/4.4).

Three artifacts per scan (written to demo/viz/; reads gt JSON from outputs/gt/):
  viz_<scan>_panel.png     - 2x3 top-down floor-plan graphs at sampled t
                             (compare stages side by side)
  viz_<scan>_floorplan.gif - top-down graph growing over t (animation)
  viz_<scan>_timeline.png  - #nodes/#edges over t + per-instance commit events

Top-down = each instance's area-weighted 3D centroid projected onto the room's
horizontal plane (the up axis is auto-detected from the 'floor' instance).
  node face color = commit_time (viridis: early=blue, late=yellow)
  node edge color = commit_reason (categorical)
  directed edges  = 3DSSG predicates (colored; arrowheads only when sparse)
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl
import mesh_instance as mi
import output as op
from paths import ROOT, DEMO_VIZ, DEFAULT_SCANS, gt_json_path, ensure_demo_dirs

ensure_demo_dirs()

REASON_COLORS = {
    "strong_single": "#e41a1c",
    "cumulative": "#377eb8",
    "persistent": "#4daf4a",
    "annotation": "#984ea3",
    "other": "#999999",
}


def load_centroids(scan_dir):
    """area-weighted 3D centroid per instance + auto-detected up axis."""
    verts, faces = dl.load_mesh(scan_dir)
    seg = dl.load_segs(scan_dir)
    s2i, labels = dl.load_semseg(scan_dir)
    face_inst, _ = mi.build_face_to_instance(faces, seg, s2i)
    face_area = mi.compute_face_areas(verts, faces)
    fc = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0
    cents = {}
    for inst in np.unique(face_inst):
        inst = int(inst)
        if inst == mi.BG:
            continue
        m = face_inst == inst
        a = face_area[m]
        s = a.sum()
        if s <= 0:
            continue
        cents[inst] = (fc[m] * a[:, None]).sum(axis=0) / s
    up = detect_up_axis(verts, seg, s2i, labels)
    return cents, up, labels


def detect_up_axis(verts, seg, s2i, labels):
    """Up = axis of minimum variance on the floor (flat surface). Fallback y-up."""
    vinst = mi.vertex_to_instance(seg, s2i)
    floor_ids = [oid for oid, lbl in labels.items()
                 if str(lbl).lower() in ("floor", "ground")]
    if floor_ids:
        fv = verts[np.isin(vinst, floor_ids)]
        if len(fv) > 10:
            return int(np.argmin(fv.var(axis=0)))
    return 1


def horiz_axes(up):
    return [a for a in (0, 1, 2) if a != up]


def _pred_palette(preds):
    preds = sorted(set(preds))
    cmap = plt.cm.tab20
    return {p: cmap(i % 20) for i, p in enumerate(preds)}


def draw_graph(ax, out, t, cents, up, labels, pred_palette, title=None):
    ax.clear()
    ha = horiz_axes(up)
    nodes, edges = op.materialize(out, t)
    T = max(1, out["num_processed_frames"] - 1)

    # background: all instance centroids (faint) for spatial context
    if cents:
        bg = np.array([c for c in cents.values()])
        ax.scatter(bg[:, ha[0]], bg[:, ha[1]], s=8, c="#e8e8e8", zorder=1)

    # edges (lines always; arrowheads only when sparse to avoid clutter)
    for e in edges:
        si, oi = int(e["subject"]), int(e["object"])
        if si in cents and oi in cents:
            p0, p1 = cents[si], cents[oi]
            col = pred_palette.get(e["predicate"], "#888888")
            if len(edges) <= 40:
                ax.annotate("", xy=(p1[ha[0]], p1[ha[1]]),
                            xytext=(p0[ha[0]], p0[ha[1]]),
                            arrowprops=dict(arrowstyle="->", color=col,
                                            alpha=0.5, lw=0.7), zorder=2)
            else:
                ax.plot([p0[ha[0]], p1[ha[0]]], [p0[ha[1]], p1[ha[1]]],
                        color=col, alpha=0.25, lw=0.5, zorder=2)

    # nodes: face=commit_time (viridis), edge=commit_reason
    if nodes:
        xs, ys, cs, es = [], [], [], []
        for nid, nd in nodes.items():
            i = int(nid)
            if i not in cents:
                continue
            c = cents[i]
            xs.append(c[ha[0]]); ys.append(c[ha[1]])
            cs.append(nd["commit_time"] / T)
            es.append(REASON_COLORS.get(nd.get("commit_reason"), "#999"))
        if xs:
            ax.scatter(xs, ys, s=110, c=cs, cmap="viridis",
                       edgecolors=es, linewidths=1.4, zorder=3,
                       vmin=0, vmax=1)
        for nid, nd in nodes.items():
            i = int(nid)
            if i not in cents:
                continue
            c = cents[i]
            ax.text(c[ha[0]], c[ha[1]], str(labels.get(i, "?"))[:9],
                    fontsize=4.5, ha="center", va="bottom", zorder=4)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def snapshot_panel(out, cents, up, labels, pred_palette, path):
    T = out["num_processed_frames"]
    ts = sorted(set([0, T // 4, T // 2, 3 * T // 4, max(0, T - 1)]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for k, t in enumerate(ts):
        ax = axes[k // 3][k % 3]
        n, e = op.materialize(out, t)
        draw_graph(ax, out, t, cents, up, labels, pred_palette,
                   title=f"t={t}  ({len(n)} nodes, {len(e)} edges)")
    # 6th cell: legend / colorbar
    lax = axes[1][2]
    lax.axis("off")
    norm = Normalize(vmin=0, vmax=max(1, T - 1))
    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=lax, fraction=0.5, pad=0.0)
    cb.set_label("commit_time (t)", fontsize=8)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaa",
                          markeredgecolor=c, markersize=9, markeredgewidth=1.6, label=r)
               for r, c in REASON_COLORS.items()]
    handles += [plt.Line2D([0], [0], color="#888", lw=1, label="directed edge (predicate)")]
    lax.legend(handles=handles, loc="center left", fontsize=7, frameon=False,
               title="node edge = commit_reason")
    fig.suptitle(f"{out['scan_id']}  top-down GT scene graph by stage", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def animate_gif(out, cents, up, labels, pred_palette, path, max_frames=60):
    T = out["num_processed_frames"]
    t_max = max(0, T - 1)
    step = max(1, T // max_frames) if T > 0 else 1
    ts = list(range(0, T, step))
    if t_max not in ts:
        ts.append(t_max)
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(i):
        t = ts[i]
        n, e = op.materialize(out, t)
        draw_graph(ax, out, t, cents, up, labels, pred_palette,
                   title=f"t={t}/{t_max}  ({len(n)} nodes, {len(e)} edges)")
        return []

    ani = FuncAnimation(fig, update, frames=len(ts), interval=250, blit=False)
    ani.save(path, writer="pillow", dpi=80)
    plt.close(fig)


def timeline(out, path):
    T = out["num_processed_frames"]
    ns, es = [], []
    for t in range(T):
        n, e = op.materialize(out, t)
        ns.append(len(n)); es.append(len(e))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.step(range(T), ns, where="post", color="#377eb8", label="nodes")
    ax1.step(range(T), es, where="post", color="#e41a1c", label="directed edges")
    ax1.set_ylabel("count in G*_{<=t}")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"{out['scan_id']}  growth over t")
    # commit events rug
    for nid, nd in out["nodes"].items():
        c = REASON_COLORS.get(nd.get("commit_reason"), "#999")
        ax2.scatter(nd["commit_time"], 0.5, color=c, s=12, marker="|")
    ax2.set_yticks([])
    ax2.set_xlabel("t (continuous frame index)")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(scan_id):
    scan_dir = os.path.join(ROOT, scan_id)
    gt_path = gt_json_path(scan_id)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"GT JSON not found: {gt_path}\nRun run.py for this scan first.")
    with open(gt_path) as f:
        out = json.load(f)
    cents, up, labels = load_centroids(scan_dir)
    preds = [e["predicate"] for e in out["edges"]]
    pred_palette = _pred_palette(preds)
    print(f"  up_axis={up} instances_with_centroid={len(cents)} predicates={len(pred_palette)}")

    p1 = os.path.join(DEMO_VIZ, f"viz_{scan_id[:8]}_panel.png")
    snapshot_panel(out, cents, up, labels, pred_palette, p1)
    print(f"  saved {p1}")

    p2 = os.path.join(DEMO_VIZ, f"viz_{scan_id[:8]}_floorplan.gif")
    animate_gif(out, cents, up, labels, pred_palette, p2)
    print(f"  saved {p2}")

    p3 = os.path.join(DEMO_VIZ, f"viz_{scan_id[:8]}_timeline.png")
    timeline(out, p3)
    print(f"  saved {p3}")


if __name__ == "__main__":
    scans = sys.argv[1:] or [DEFAULT_SCANS[0]]
    for sc in scans:
        print(f"\n=== viz {sc} ===")
        run(sc)
