"""Read one 3RScan scan: mesh geometry, per-vertex instance labels, camera
stream (pose + intrinsics), and the 3DSSG relationships for that scan.

Verified facts about this dataset layout (checked on the sample scans):
  - mesh.refined.v2.obj, labels.instances.annotated.v2.ply and segIndices all
    share the SAME vertex order/count (18496 verts, 26517 faces on the sample).
  - The PLY carries a per-vertex `objectId`, which is identical (100%) to the
    instance derived via semseg.segGroups -> segments -> segIndices. We use the
    semseg+segs path as the primary source (per handoff doc) and keep the PLY
    objectId for cross-check.
  - relationships.json entries are keyed by scan; each relation row is
    [subjectId, objectId, predId, predName].
"""
import json
import os
import zipfile
import numpy as np
import trimesh


def load_mesh(scan_dir):
    """Load OBJ keeping original vertex/face order (process=False is critical:
    it must stay aligned with segIndices and the PLY)."""
    obj_path = os.path.join(scan_dir, "mesh.refined.v2.obj")
    mesh = trimesh.load(obj_path, process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


def load_segs(scan_dir):
    """segIndices: per-vertex over-segmentation id (len == num vertices)."""
    p = os.path.join(scan_dir, "mesh.refined.0.010000.segs.v2.json")
    d = json.load(open(p))
    return np.asarray(d["segIndices"], dtype=np.int64)


def load_semseg(scan_dir):
    """Return (seg_to_instance: dict segId->objId, instance_label: dict objId->label)."""
    p = os.path.join(scan_dir, "semseg.v2.json")
    d = json.load(open(p))
    seg_to_instance = {}
    instance_label = {}
    for g in d["segGroups"]:
        oid = g["objectId"]
        instance_label[oid] = g["label"]
        for s in g["segments"]:
            seg_to_instance[s] = oid
    return seg_to_instance, instance_label


def load_ply_objectid(scan_dir, n_verts):
    """Per-vertex objectId straight from the annotated PLY (cross-check source)."""
    p = os.path.join(scan_dir, "labels.instances.annotated.v2.ply")
    with open(p, "rb") as f:
        raw = f.read()
    body = raw.split(b"end_header\n", 1)[1].decode("latin1").splitlines()
    # columns: x y z r g b objectId globalId NYU40 Eigen13 RIO27
    obj = np.array([int(line.split()[6]) for line in body[:n_verts]], dtype=np.int64)
    return obj


def load_intrinsics(scan_dir):
    """Read color intrinsics K (3x3) and resolution from sequence/_info.txt."""
    info_path = os.path.join(scan_dir, "sequence", "_info.txt")
    if not os.path.exists(info_path):
        # fall back to reading from the zip
        with zipfile.ZipFile(os.path.join(scan_dir, "sequence.zip")) as z:
            txt = z.read("_info.txt").decode()
    else:
        txt = open(info_path).read()
    info = {}
    for line in txt.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    vals = [float(x) for x in info["m_calibrationColorIntrinsic"].split()]
    K = np.array(vals, dtype=np.float64).reshape(4, 4)[:3, :3]
    W = int(info["m_colorWidth"])
    H = int(info["m_colorHeight"])
    n = int(info["m_frames.size"])
    return K, W, H, n


def iter_frames(scan_dir):
    """Yield (frame_id_str, pose 4x4 cam->world) in ascending frame order.
    Frames with missing/invalid pose are skipped here (caller re-indexes t)."""
    seq = os.path.join(scan_dir, "sequence")
    pose_files = sorted(f for f in os.listdir(seq) if f.endswith(".pose.txt"))
    for pf in pose_files:
        fid = pf.replace(".pose.txt", "")
        pose = np.loadtxt(os.path.join(seq, pf), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            continue
        # 3RScan marks invalid poses with -inf; isfinite check above handles it.
        yield fid, pose


def load_relationships(dataset_root, scan_id, split=None):
    """Find this scan's relationship rows. Try the split file first if given,
    else fall back to the full relationships.json. Returns list of
    (subjectId, objectId, predId, predName)."""
    ssg = os.path.join(dataset_root, "3DSSG_subset")
    candidates = []
    if split:
        candidates.append(f"relationships_{split}.json")
    candidates += ["relationships.json", "relationships_train.json",
                   "relationships_validation.json", "relationships_test.json"]
    for fn in candidates:
        fp = os.path.join(ssg, fn)
        if not os.path.exists(fp):
            continue
        data = json.load(open(fp))
        for s in data["scans"]:
            if s["scan"] == scan_id:
                return [tuple(r) for r in s["relationships"]], fn
    return [], None
