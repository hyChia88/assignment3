import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        if ray_bundle.origins.dim() == 3:
            ray_bundle = ray_bundle._replace(
                origins=ray_bundle.origins.reshape(-1, 3),
                directions=ray_bundle.directions.reshape(-1, 3),
            )
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        B = ray_bundle.origins.shape[0]  # 射线条数 R（已展平）
        # print("[DEBUG] B", B) 
        device = ray_bundle.origins.device
        dtype = ray_bundle.origins.dtype
        base = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray,
                              device=device, dtype=dtype)                    # (n,)
        z_vals = base.unsqueeze(0).expand(B, -1)                              # (R, n)
        # print("[DEBUG] z_vals", z_vals.shape)
        with torch.no_grad():
            mids  = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])                   # (R, n-1)
            lower = torch.cat([z_vals[:, :1], mids], dim=-1)                 # (R, n)
            upper = torch.cat([mids, z_vals[:, -1:].clone()], dim=-1)        # (R, n)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand                        # (R, n)

        # TODO (Q1.4): Sample points from z values
        o = ray_bundle.origins.unsqueeze(1)                                   # (R, 1, 3)
        d = ray_bundle.directions.unsqueeze(1)                                # (R, 1, 3)
        # print("[DEBUG] o", o.shape)
        # print("[DEBUG] d", d.shape)
        # print("[DEBUG] ray_bundle", ray_bundle.shape)
        sample_points = o + z_vals.unsqueeze(-1) * d                          # (R, n, 3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals.unsqueeze(-1).expand_as(sample_points[..., :1]),  # (R, n, 1)
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}