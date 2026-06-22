"""Smoke test for the rendering core (highest-risk unknown).

Loads one scan, builds face_to_instance, renders a few frames, overlays the
instance-id buffer onto the RGB frame, and reports visible instances. This
validates: mesh/seg alignment, occlusion, coordinate/units, and the renderer.
"""
import io
import os
import sys
import zipfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl
import mesh_instance as mi
from renderer import FaceRenderer

ROOT = "/home/data16t1/fengchangqun/AgentSGG/3RScan"
SCAN = "7272e16c-a01b-20f6-8961-a0927b4a7629"
OUT = os.path.join("/home/data16t1/fengchangqun/AgentSGG", "gt_subgraph", "out")
os.makedirs(OUT, exist_ok=True)


def color_for(inst):
    rng = np.random.RandomState(inst * 9973 + 7)
    return rng.randint(60, 256, size=3)


def load_color(scan_dir, fid):
    rgb_path = os.path.join(scan_dir, "sequence", f"{fid}.color.jpg")
    if os.path.exists(rgb_path):
        return Image.open(rgb_path).convert("RGB")
    with zipfile.ZipFile(os.path.join(scan_dir, "sequence.zip")) as z:
        names = [n for n in z.namelist() if os.path.basename(n) == f"{fid}.color.jpg"]
        if not names:
            raise FileNotFoundError(f"{fid}.color.jpg not found in sequence or sequence.zip")
        return Image.open(io.BytesIO(z.read(names[0]))).convert("RGB")


def main():
    scan_dir = os.path.join(ROOT, SCAN)
    verts, faces = dl.load_mesh(scan_dir)
    seg_indices = dl.load_segs(scan_dir)
    seg_to_instance, instance_label = dl.load_semseg(scan_dir)
    print(f"verts={len(verts)} faces={len(faces)} segIdx={len(seg_indices)}")
    assert len(verts) == len(seg_indices), "vertex/seg count mismatch!"

    face_inst, n_dis = mi.build_face_to_instance(faces, seg_indices, seg_to_instance)
    face_area = mi.compute_face_areas(verts, faces)
    tot_area = mi.total_area_per_instance(face_inst, face_area)
    ply = dl.load_ply_objectid(scan_dir, len(verts))
    dis_frac = mi.crosscheck_with_ply(face_inst, faces, ply)
    print(f"face_to_instance: {len(tot_area)} instances, "
          f"all-3-differ faces={n_dis}, PLY-crosscheck disagree={dis_frac:.4%}")

    K, W, H, n = dl.load_intrinsics(scan_dir)
    print(f"intrinsics K=\n{K}\nres={W}x{H} declared_frames={n}")

    rend = FaceRenderer(verts, faces)

    frames = list(dl.iter_frames(scan_dir))
    print(f"valid-pose frames: {len(frames)}")
    pick = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    for t in pick:
        fid, pose = frames[t]
        prim = rend.render_face_ids(K, pose, W, H)
        hit = prim >= 0
        inst_buf = np.full((H, W), mi.BG, dtype=np.int64)
        inst_buf[hit] = face_inst[prim[hit]]
        vis, counts = np.unique(inst_buf[inst_buf != mi.BG], return_counts=True)
        order = np.argsort(-counts)
        top = [(int(vis[i]), instance_label.get(int(vis[i]), "?"), int(counts[i]))
               for i in order[:8]]
        print(f"\n[{fid}] hit-pixels={hit.sum()}/{H*W} visible_instances={len(vis)}")
        print("  top:", top)

        # overlay
        rgb = np.array(load_color(scan_dir, fid).resize((W, H)))
        overlay = rgb.copy()
        for inst in vis:
            overlay[inst_buf == inst] = color_for(int(inst))
        blend = (0.5 * rgb + 0.5 * overlay).astype(np.uint8)
        Image.fromarray(blend).save(os.path.join(OUT, f"overlay_{fid}.png"))
    print(f"\nOverlays written to {OUT}")


if __name__ == "__main__":
    main()
