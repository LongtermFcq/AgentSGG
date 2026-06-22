"""Render all frames of a scan and dump the distributions the handoff doc
requires for choosing thresholds (Section 4.2). Does NOT commit with final
thresholds yet -- it collects raw evidence so we pick TAU_* from the data."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl
import mesh_instance as mi
import gt_builder as gb

ROOT = "/home/data16t1/fengchangqun/AgentSGG/3RScan"
OUT = os.path.join("/home/data16t1/fengchangqun/AgentSGG", "gt_subgraph", "out")


def run(scan_id):
    scan_dir = os.path.join(ROOT, scan_id)
    verts, faces = dl.load_mesh(scan_dir)
    seg_indices = dl.load_segs(scan_dir)
    seg_to_instance, labels = dl.load_semseg(scan_dir)
    face_inst, _ = mi.build_face_to_instance(faces, seg_indices, seg_to_instance)
    face_area = mi.compute_face_areas(verts, faces)
    total_area = mi.total_area_per_instance(face_inst, face_area)
    K, W, H, _ = dl.load_intrinsics(scan_dir)
    frames = list(dl.iter_frames(scan_dir))

    # collect-stats pass with permissive thresholds so we record everything
    cfg = gb.Config(TAU_INST_PIX=1, TAU_FACE_PIX=2, TAU_STRONG=2.0,
                    TAU_COMMIT=2.0, ENABLE_PERSIST=False)
    res = gb.build_scan(verts, faces, face_inst, face_area, total_area,
                        frames, K, W, H, cfg, collect_stats=True)
    s = res["stats"]
    single = np.array(s["single"])
    cum = np.array(s["cumulative"])
    inst_pix = np.array(s["inst_pix"])

    def q(a, ps):
        if a.size == 0:
            return {p: None for p in ps}
        return {p: float(np.percentile(a, p)) for p in ps}

    report = {
        "scan_id": scan_id,
        "n_frames": len(frames),
        "n_instances": len(total_area),
        "single_area_ratio": {
            "n": int(single.size),
            "max": float(single.max()) if single.size else None,
            "pct": q(single, [50, 75, 90, 95, 99]),
        },
        "cumulative_area_ratio": {
            "n": int(cum.size),
            "max": float(cum.max()) if cum.size else None,
            "pct": q(cum, [50, 75, 90, 95, 99]),
        },
        "inst_pixels_per_frame": {
            "n": int(inst_pix.size),
            "pct": q(inst_pix, [5, 10, 25, 50, 75, 90]),
        },
    }
    print(json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(single, bins=50); ax[0].set_title("single_area_ratio")
    ax[1].hist(cum, bins=50); ax[1].set_title("cumulative_area_ratio")
    ax[2].hist(np.log10(inst_pix + 1), bins=50); ax[2].set_title("log10 inst pixels/frame")
    fig.suptitle(f"{scan_id}  ({len(total_area)} inst, {len(frames)} frames)")
    fig.tight_layout()
    p = os.path.join(OUT, f"stats_{scan_id[:8]}.png")
    fig.savefig(p, dpi=90)
    with open(os.path.join(OUT, f"stats_{scan_id[:8]}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("saved", p)


if __name__ == "__main__":
    scans = sys.argv[1:] or ["7272e16c-a01b-20f6-8961-a0927b4a7629"]
    for sc in scans:
        run(sc)
