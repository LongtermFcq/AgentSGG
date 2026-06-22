"""Phase 0: build face_to_instance + areas + ID cross-checks (frame-independent)."""
import numpy as np

BG = 0  # objectId 0 = background / unlabeled in the PLY & our convention


def build_face_to_instance(faces, seg_indices, seg_to_instance):
    """vertex -> segment (seg_indices) -> instance (seg_to_instance), then a face
    takes the majority instance of its 3 vertices. Returns:
      face_to_instance (n_faces,) int64  (BG=0 where unlabeled)
      n_disagree: faces whose 3 verts disagreed (debug stat)
    """
    vert_inst = np.array(
        [seg_to_instance.get(int(s), BG) for s in seg_indices], dtype=np.int64
    )
    fv = vert_inst[faces]  # (n_faces, 3)
    # majority vote per row
    face_inst = np.empty(len(faces), dtype=np.int64)
    n_disagree = 0
    same01 = fv[:, 0] == fv[:, 1]
    same12 = fv[:, 1] == fv[:, 2]
    same02 = fv[:, 0] == fv[:, 2]
    all_same = same01 & same12
    face_inst[all_same] = fv[all_same, 0]
    rest = ~all_same
    for idx in np.nonzero(rest)[0]:
        a, b, c = fv[idx]
        if a == b or a == c:
            face_inst[idx] = a
        elif b == c:
            face_inst[idx] = b
        else:
            # all three differ -> pick vertex[0] deterministically
            face_inst[idx] = a
            n_disagree += 1
    return face_inst, n_disagree


def compute_face_areas(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def total_area_per_instance(face_inst, face_area):
    out = {}
    for inst in np.unique(face_inst):
        if inst == BG:
            continue
        out[int(inst)] = float(face_area[face_inst == inst].sum())
    return out


def crosscheck_with_ply(face_inst, faces, ply_objid):
    """Compare our face_to_instance against a face label derived from the PLY's
    per-vertex objectId (majority). Returns disagreement fraction over non-BG."""
    fv = ply_objid[faces]
    # majority per face for the PLY
    ply_face = fv[:, 0].copy()
    m12 = fv[:, 1] == fv[:, 2]
    ply_face[m12] = fv[m12, 1]  # if v1==v2 use that; else keep v0 (rough)
    mask = (face_inst != BG) | (ply_face != BG)
    disagree = np.sum(face_inst[mask] != ply_face[mask])
    return disagree / max(mask.sum(), 1)


def vertex_to_instance(seg_indices, seg_to_instance):
    """Per-vertex instance id via the semseg->segs->vertex chain. BG where unmapped."""
    return np.array(
        [seg_to_instance.get(int(s), BG) for s in seg_indices], dtype=np.int64
    )


def instance_diagnostics(face_inst, face_area, seg_indices, seg_to_instance,
                         faces, instance_label):
    """Per-instance geometry coverage for ALL semseg instances (including ones
    with no mesh faces). For each objectId report:
      label, vertex_count (via semseg chain), vertex_referenced (subset that is
      actually used by >=1 face), face_count, total_area.

    This separates three cases that total_area_per_instance alone conflates:
      - normal:       has faces + area  -> renderable, committable
      - annotation-only: vertices exist but none referenced by faces -> no
                        surface, can NEVER be rendered/committed (common in
                        3RScan for lamps/lights/small items the mesh didn't
                        reconstruct). NOT a bug.
      - truly empty:  no vertices at all
    """
    vinst = vertex_to_instance(seg_indices, seg_to_instance)
    referenced = np.zeros(len(seg_indices), dtype=bool)
    referenced[faces.ravel()] = True
    rows = {}
    for oid in sorted(instance_label.keys()):
        vmask = vinst == oid
        rows[int(oid)] = {
            "label": instance_label[oid],
            "vertex_count": int(vmask.sum()),
            "vertex_referenced": int((vmask & referenced).sum()),
            "face_count": int((face_inst == oid).sum()),
            "total_area": float(face_area[face_inst == oid].sum()),
        }
    return rows


def summarize_diagnostics(rows):
    """Roll up instance_diagnostics rows into counts for a quick log line."""
    n = len(rows)
    has_geom = sum(1 for r in rows.values() if r["face_count"] > 0)
    annot_only = sum(1 for r in rows.values()
                     if r["face_count"] == 0 and r["vertex_count"] > 0)
    empty = sum(1 for r in rows.values()
                if r["face_count"] == 0 and r["vertex_count"] == 0)
    return {
        "n_instances": n, "has_geometry": has_geom,
        "annotation_only": annot_only, "empty": empty,
    }
