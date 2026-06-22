"""Render a per-pixel face-ID buffer with z-buffer occlusion via Open3D
RaycastingScene. cast_rays returns `primitive_ids` (= triangle index of the
closest hit per pixel) which is exactly the face-ID buffer Phase A needs.

Coordinate convention:
  - pose.txt is camera->world (P_t). RaycastingScene wants world->camera as the
    extrinsic, so we pass inv(P_t).
  - create_rays_pinhole uses the OpenCV-style pinhole (x right, y down,
    +z forward), which matches the 3RScan tango calibration.

Two renderers share the EXACT same ray generation (`_make_rays`) so that
pix_vis (full-scene render) and pix_full (instance-only render) differ only in
which triangles are in the scene -- never in camera/ray setup. This is what
makes TAU_INST_VIS_RATIO = pix_vis / pix_full a clean occlusion-normalized
ratio with no extra inconsistency source.
"""
import numpy as np
import open3d as o3d

INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID  # sentinel for "ray hit nothing"


def _make_rays(K, pose_cam2world, W, H):
    """Identical ray bundle for both renderers: world->camera extrinsic = inv(P_t)."""
    extrinsic = np.linalg.inv(pose_cam2world)  # world -> camera
    return o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float32),
        extrinsic_matrix=o3d.core.Tensor(extrinsic, dtype=o3d.core.Dtype.Float32),
        width_px=int(W), height_px=int(H),
    )


class FaceRenderer:
    def __init__(self, verts, faces):
        self.scene = o3d.t.geometry.RaycastingScene()
        v = o3d.core.Tensor(np.ascontiguousarray(verts), dtype=o3d.core.Dtype.Float32)
        f = o3d.core.Tensor(np.ascontiguousarray(faces), dtype=o3d.core.Dtype.UInt32)
        self.scene.add_triangles(v, f)
        self.n_faces = len(faces)

    def render_face_ids(self, K, pose_cam2world, W, H):
        """Return an (H, W) int64 array of face ids; -1 where no hit."""
        rays = _make_rays(K, pose_cam2world, W, H)
        prim = self.scene.cast_rays(rays)["primitive_ids"].numpy()
        out = prim.astype(np.int64)
        out[prim == INVALID] = -1
        return out


class InstanceRenderer:
    """One RaycastingScene per instance, each containing ONLY that instance's
    triangles. Used to compute pix_full = the instance's complete (unoccluded)
    projection under the SAME camera as the main render. No bbox / 2D-area
    approximation: the only difference from FaceRenderer is the triangle subset,
    so pix_vis / pix_full is a pure occlusion-normalized ratio.

    Built once at Phase 0. Query per (instance, frame) -- only for instances
    actually touched and not yet committed, to keep cost bounded.
    """
    def __init__(self, verts, faces, face_inst, bg=0):
        self.scenes = {}
        for inst in np.unique(face_inst):
            inst = int(inst)
            if inst == bg:
                continue
            fidx = np.nonzero(face_inst == inst)[0]
            sc = o3d.t.geometry.RaycastingScene()
            v = o3d.core.Tensor(np.ascontiguousarray(verts), dtype=o3d.core.Dtype.Float32)
            f = o3d.core.Tensor(np.ascontiguousarray(faces[fidx]), dtype=o3d.core.Dtype.UInt32)
            sc.add_triangles(v, f)
            self.scenes[inst] = sc

    def pix_full(self, inst, K, pose_cam2world, W, H):
        """Number of pixels where this instance's own triangles are hit when
        rendered alone (its full silhouette, incl. self-occlusion, no other
        objects). 0 if the instance has no scene (e.g. annotation-only)."""
        sc = self.scenes.get(int(inst))
        if sc is None:
            return 0
        rays = _make_rays(K, pose_cam2world, W, H)
        prim = sc.cast_rays(rays)["primitive_ids"].numpy()
        return int(np.count_nonzero(prim != INVALID))
