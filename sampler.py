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

# 4. NeRF Extras (CHOOSE ONE! More than one is extra credit)
class HierarchicalSampler(torch.nn.Module):
    """
    Hierarchical sampler for NeRF coarse-to-fine sampling.
    Takes coarse samples and their weights, then samples additional
    points in regions with high density (importance sampling).
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(self, ray_bundle, coarse_weights=None):
        """
        Args:
            ray_bundle: RayBundle with origins and directions
            coarse_weights: Optional weights from coarse network for importance sampling

        Returns:
            ray_bundle with sampled points (combined coarse + fine if weights provided)
        """
        if ray_bundle.origins.dim() == 3:
            ray_bundle = ray_bundle._replace(
                origins=ray_bundle.origins.reshape(-1, 3),
                directions=ray_bundle.directions.reshape(-1, 3),
            )

        B = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        dtype = ray_bundle.origins.dtype

        if coarse_weights is None:
            # Initial stratified sampling (same as StratifiedRaysampler)
            base = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray,
                                  device=device, dtype=dtype)
            z_vals = base.unsqueeze(0).expand(B, -1)

            with torch.no_grad():
                mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
                lower = torch.cat([z_vals[:, :1], mids], dim=-1)
                upper = torch.cat([mids, z_vals[:, -1:].clone()], dim=-1)
                t_rand = torch.rand_like(z_vals)
                z_vals = lower + (upper - lower) * t_rand
        else:
            # Importance sampling based on coarse weights
            # Get coarse z_vals from ray_bundle
            z_vals_coarse = ray_bundle.sample_lengths[..., 0]  # (B, N_coarse)

            # Perform hierarchical sampling
            z_vals_fine = self.sample_pdf(
                z_vals_coarse,
                coarse_weights,
                self.n_pts_per_ray,
                det=False
            )

            # Combine coarse and fine samples and sort
            z_vals, _ = torch.sort(
                torch.cat([z_vals_coarse, z_vals_fine], dim=-1),
                dim=-1
            )

        # Sample points from z values
        o = ray_bundle.origins.unsqueeze(1)  # (B, 1, 3)
        d = ray_bundle.directions.unsqueeze(1)  # (B, 1, 3)
        sample_points = o + z_vals.unsqueeze(-1) * d  # (B, N, 3)

        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals.unsqueeze(-1).expand_as(sample_points[..., :1]),
        )

    def sample_pdf(self, bins, weights, N_samples, det=False):
        """
        Sample additional points along rays according to probability density function.

        Args:
            bins: (B, N_coarse) z-values of coarse samples
            weights: (B, N_coarse, 1) or (B, N_coarse-1) weights from coarse network
            N_samples: number of fine samples to draw
            det: deterministic sampling if True

        Returns:
            z_samples: (B, N_samples) new z-values
        """
        device = bins.device
        dtype = bins.dtype

        # Handle weight shape - could be (B, N, 1) or (B, N-1)
        if weights.dim() == 3:
            weights = weights.squeeze(-1)  # (B, N, 1) -> (B, N)

        # Weights should be (B, N-1) for bins between samples
        # If weights.shape[1] == bins.shape[1], we need to convert to mid-point weights
        if weights.shape[1] == bins.shape[1]:
            # Take midpoint weights (average adjacent weights)
            weights = 0.5 * (weights[:, :-1] + weights[:, 1:])  # (B, N-1)

        # Prevent nans
        weights = weights + 1e-5

        # Get pdf (normalize weights)
        pdf = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)  # (B, N-1)
        cdf = torch.cumsum(pdf, dim=-1)  # (B, N-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (B, N)

        # Take uniform samples
        B = bins.shape[0]
        if det:
            u = torch.linspace(0., 1., N_samples, device=device, dtype=dtype)
            u = u.unsqueeze(0).expand(B, -1)
        else:
            u = torch.rand(B, N_samples, device=device, dtype=dtype)

        # Invert CDF to get sample positions
        u = u.contiguous()

        # Avoid edge cases
        u = torch.clamp(u, 0.0 + 1e-5, 1.0 - 1e-5)

        indices = torch.searchsorted(cdf.contiguous(), u, right=True)  # (B, N_samples)
        below = torch.clamp(indices - 1, min=0)
        above = torch.clamp(indices, max=cdf.shape[-1] - 1)

        # Ensure indices are valid
        indices_g = torch.stack([below, above], dim=-1)  # (B, N_samples, 2)
        indices_g = torch.clamp(indices_g, 0, bins.shape[-1] - 1)

        # Gather CDF and bin values using proper indexing
        # Need to be careful with dimensions for gather
        B, N_bins = bins.shape
        _, N_cdf = cdf.shape

        # Expand and gather
        matched_shape = [B, N_samples, N_cdf]
        cdf_expanded = cdf.unsqueeze(1).expand(matched_shape)
        cdf_g = torch.gather(cdf_expanded, 2, indices_g)  # (B, N_samples, 2)

        matched_shape_bins = [B, N_samples, N_bins]
        bins_expanded = bins.unsqueeze(1).expand(matched_shape_bins)
        bins_g = torch.gather(bins_expanded, 2, indices_g)  # (B, N_samples, 2)

        # Linear interpolation
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples


sampler_dict = {
    'stratified': StratifiedRaysampler,
    'hierarchical': HierarchicalSampler
}