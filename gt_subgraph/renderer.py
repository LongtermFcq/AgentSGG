"""Render a per-pixel face-ID buffer with z-buffer occlusion via Open3D
RaycastingScene. cast_rays returns `primitive_ids` (= triangle index of the
closest hit per pixel) which is exactly the face-ID buffer Phase A needs.

Coordinate convention:
  - pose.txt is camera->world (P_t). RaycastingScene wants world->camera as the
    extrinsic, so we pass inv(P_t).
  - create_rays_pinhole uses the OpenCV-style pinhole (x right, y down,
    +z forward), which matches the 3RScan tango calibration.
"""
import numpy as np
import open3d as o3d

INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID  # sentinel for "ray hit nothing"


class FaceRenderer:
    def __init__(self, verts, faces):
        self.scene = o3d.t.geometry.RaycastingScene()
        v = o3d.core.Tensor(np.ascontiguousarray(verts), dtype=o3d.core.Dtype.Float32)
        f = o3d.core.Tensor(np.ascontiguousarray(faces), dtype=o3d.core.Dtype.UInt32)
        self.scene.add_triangles(v, f)
        self.n_faces = len(faces)

    def render_face_ids(self, K, pose_cam2world, W, H):
        """Return an (H, W) int64 array of face ids; INVALID where no hit."""
        extrinsic = np.linalg.inv(pose_cam2world)  # world -> camera
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float32),
            extrinsic_matrix=o3d.core.Tensor(extrinsic, dtype=o3d.core.Dtype.Float32),
            width_px=int(W), height_px=int(H),
        )
        ans = self.scene.cast_rays(rays)
        prim = ans["primitive_ids"].numpy().astype(np.int64)  # (H, W), UInt32 INVALID -> large
        # normalize the INVALID sentinel to -1
        prim[ans["primitive_ids"].numpy() == INVALID] = -1
        return prim
