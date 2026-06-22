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
import io
import os
import warnings
import zipfile
import numpy as np


def load_mesh(scan_dir):
    """Parse the OBJ manually, preserving EXACTLY its vertex list and order.

    Do NOT use trimesh/open3d loaders here: they may drop/merge vertices (e.g.
    trimesh dropped 4 of 43877 verts on one sample), which silently breaks
    alignment with segIndices (per-vertex) and the PLY. We keep all `v` lines in
    order and read the first (geometry) index of each `f` corner (1-based)."""
    obj_path = os.path.join(scan_dir, "mesh.refined.v2.obj")
    verts = []
    faces = []
    n_ngon = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line[2:].split()[:3])
            elif line.startswith("f "):
                parts = line[2:].split()
                # f v/vt/vn ... -> take geometry index (before first '/')
                indices = [p.split("/", 1)[0] for p in parts]
                if len(indices) < 3:
                    continue
                if len(indices) > 3:
                    n_ngon += 1
                # Fan-triangulate quads/n-gons; downstream expects triangles.
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])
    if n_ngon:
        warnings.warn(
            f"Triangulated {n_ngon} non-triangular face(s) in {obj_path}",
            stacklevel=2,
        )
    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64) - 1  # OBJ is 1-indexed
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
        # Fall back to reading from the zip. Some archives store files at the
        # root, others under a sequence/ prefix.
        with zipfile.ZipFile(os.path.join(scan_dir, "sequence.zip")) as z:
            info_names = [n for n in z.namelist() if os.path.basename(n) == "_info.txt"]
            if not info_names:
                raise FileNotFoundError(f"_info.txt not found in {scan_dir}/sequence.zip")
            txt = z.read(info_names[0]).decode()
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
    if os.path.isdir(seq):
        pose_files = sorted(f for f in os.listdir(seq) if f.endswith(".pose.txt"))
        for pf in pose_files:
            fid = pf.replace(".pose.txt", "")
            pose = np.loadtxt(os.path.join(seq, pf), dtype=np.float64)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                continue
            # 3RScan marks invalid poses with -inf; isfinite check above handles it.
            yield fid, pose
        return

    zip_path = os.path.join(scan_dir, "sequence.zip")
    with zipfile.ZipFile(zip_path) as z:
        pose_names = sorted(
            n for n in z.namelist() if os.path.basename(n).endswith(".pose.txt")
        )
        for name in pose_names:
            fid = os.path.basename(name).replace(".pose.txt", "")
            txt = z.read(name).decode()
            pose = np.loadtxt(io.StringIO(txt), dtype=np.float64)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                continue
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


def validate_invariants(verts, faces, seg_indices, ply_objid, scan_id=""):
    """Loader-level invariants (Phase 0 prerequisite). Hard-fails on violations
    that would silently corrupt face_to_instance; returns a stats dict otherwise.

    Checks (per handoff doc):
      (a) raw_obj_vertex_count == len(segIndices) == ply_vertex_count
      (b) max(face_index) < raw_obj_vertex_count  (and min >= 0)
      (c) num_faces reasonable (non-zero, not absurdly large vs verts)
    """
    n_v = len(verts)
    n_seg = len(seg_indices)
    n_ply = len(ply_objid)
    n_f = len(faces)
    stats = {"obj_vertices": n_v, "seg_indices": n_seg,
             "ply_vertices": n_ply, "faces": n_f}
    # (a) three vertex counts must agree
    if not (n_v == n_seg == n_ply):
        raise ValueError(
            f"[{scan_id}] vertex count mismatch: obj={n_v} seg={n_seg} ply={n_ply} "
            "-- face_to_instance alignment would be corrupted; aborting.")
    # (b) face indices in range [0, n_v)
    if n_f == 0:
        raise ValueError(f"[{scan_id}] parsed 0 faces -- OBJ has no 'f' lines?")
    fmin = int(faces.min())
    fmax = int(faces.max())
    if fmin < 0 or fmax >= n_v:
        raise ValueError(
            f"[{scan_id}] face index out of range: min={fmin} max={fmax} nverts={n_v}")
    stats["face_index_min"] = fmin
    stats["face_index_max"] = fmax
    # (c) reasonable face count: triangles should be a sane multiple of verts.
    # 3RScan refined meshes have many unreferenced annotation verts, so faces can
    # be < verts; flag only gross anomalies (e.g. >8x verts or zero).
    ratio = n_f / max(n_v, 1)
    stats["faces_per_vert"] = round(ratio, 3)
    if ratio > 8.0:
        warnings.warn(
            f"[{scan_id}] suspiciously many faces: {n_f} faces / {n_v} verts "
            f"(ratio={ratio:.2f}) -- check OBJ parsing",
            stacklevel=2)
    # how many verts are actually referenced by >=1 face (annotation-only verts
    # are expected in 3RScan but worth reporting).
    referenced = np.zeros(n_v, dtype=bool)
    referenced[faces.ravel()] = True
    stats["verts_referenced"] = int(referenced.sum())
    stats["verts_unreferenced"] = int((~referenced).sum())
    return stats
