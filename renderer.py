import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        alpha = 1.0 - torch.exp(-rays_density * deltas)
        transmittance = torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1, 1), device=alpha.device), (1.0 - alpha + eps)],
                dim=1,
            ),
            dim=1,
        )[:, :-1]

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = transmittance * alpha
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def _render_with_implicit(self, ray_bundle, implicit_fn, n_pts):
        """
        Helper method to render a ray bundle with an implicit function.
        Used for both coarse and fine passes in hierarchical sampling.

        Args:
            ray_bundle: RayBundle with sampled points
            implicit_fn: Implicit function (NeRF network)
            n_pts: Number of points per ray

        Returns:
            dict with 'feature', 'depth', and 'weights' keys
        """
        # Call implicit function with sample points
        implicit_output = implicit_fn(ray_bundle)
        density = implicit_output['density']
        feature = implicit_output['feature']

        # Compute length of each ray segment
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        # Compute aggregation weights
        weights = self._compute_weights(
            deltas.view(-1, n_pts, 1),
            density.view(-1, n_pts, 1)
        )

        # Render (color) features using weights
        color = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))

        # Render depth map
        depth = torch.sum(weights * depth_values.view(-1, n_pts, 1), dim=1)

        return {
            'feature': color,
            'depth': depth,
            'weights': weights,  # Return weights for hierarchical sampling
        }

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Render using helper method
            cur_out = self._render_with_implicit(cur_ray_bundle, implicit_fn, n_pts)

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Sphere tracing algorithm for rendering SDFs.

        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        N = origins.shape[0]
        device = origins.device

        # Initialize ray marching
        # Start at near plane and march along ray direction
        t = torch.ones(N, 1, device=device) * self.near  # (N, 1) distance along ray
        points = origins + t * directions  # (N, 3) current positions

        # Mask tracking which rays have hit the surface
        hit_mask = torch.zeros(N, 1, device=device, dtype=torch.bool)  # (N, 1)

        # Sphere tracing loop
        for i in range(self.max_iters):
            # Query SDF at current points
            sdf_values = implicit_fn(points)  # (N, 1)

            # Check which rays are close enough to surface (hit condition)
            # Threshold determines how close we need to be to consider it a "hit"
            surface_threshold = 1e-3
            newly_hit = (torch.abs(sdf_values) < surface_threshold)  # (N, 1)

            # Update hit mask (once hit, always hit)
            hit_mask = hit_mask | newly_hit

            # March forward by the SDF distance (only for rays that haven't hit yet)
            # Rays that already hit should stop marching
            active_mask = ~hit_mask  # Rays still marching
            t = t + sdf_values * active_mask.float()  # Only update active rays

            # Check if we've marched beyond far plane (missed the surface)
            t = torch.clamp(t, self.near, self.far)

            # Update positions
            points = origins + t * directions

            # Early termination: if all rays have hit or exceeded far plane
            if hit_mask.all():
                break

        # Final mask indicates which rays successfully hit the surface
        mask = hit_mask

        return points, mask

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            # 兜底：把 (B_cam, N, 3) 统一为 (R, 3)
            if cur_ray_bundle.origins.dim() == 3:
                cur_ray_bundle = cur_ray_bundle._replace(
                    origins=cur_ray_bundle.origins.reshape(-1, 3),
                    directions=cur_ray_bundle.directions.reshape(-1, 3),
                )

            # Sphere tracing already computed intersection points, no sampling needed
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    """
    Convert SDF to density using Laplace CDF (VolSDF paper Section 3.1)

    The Laplace CDF is: Φ(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x|/β))
    Density: σ(s) = α * Φ(-s/β)

    Intuition:
    - alpha: controls overall density magnitude (higher = more opaque)
    - beta: controls transition sharpness near surface (lower = sharper)
    """
    # Apply Laplace CDF to negative SDF
    psi = torch.sigmoid(-signed_distance / beta)
    return alpha * psi

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = None # TODO (Q7): convert SDF to density
            density = sdf_to_density(distance, self.alpha, self.beta)

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
