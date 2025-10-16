import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        # Get configuration parameters
        n_layers = cfg.n_layers_xyz
        hidden_dim = cfg.n_hidden_neurons_xyz
        append_xyz = cfg.append_xyz if hasattr(cfg, 'append_xyz') else []

        # Build MLP for processing position embeddings
        # Use MLPWithInputSkips if skip connections are specified
        if append_xyz:
            self.mlp_xyz = MLPWithInputSkips(
                n_layers=n_layers,
                input_dim=embedding_dim_xyz,
                output_dim=hidden_dim,
                skip_dim=embedding_dim_xyz,
                hidden_dim=hidden_dim,
                input_skips=append_xyz
            )
        else:
            # Build simple MLP without skip connections
            layers = []
            for i in range(n_layers):
                if i == 0:
                    layers.append(torch.nn.Linear(embedding_dim_xyz, hidden_dim))
                else:
                    layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(torch.nn.ReLU(True))
            self.mlp_xyz = torch.nn.Sequential(*layers)

        # Output layers
        # Density head: outputs a single scalar for density
        self.density_layer = torch.nn.Linear(hidden_dim, 1)

        # Color head: outputs 3 values for RGB color
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dim, 3)
        )

        # Store config for potential use
        self.cfg = cfg
        self.append_xyz = append_xyz
    
    # def positional_encoding(self, x):
    #     # x: (..., 3)
    #     if not self.use_pe:
    #         return x
    #     freqs = 2.0 ** torch.arange(self.num_freqs, device=x.device, dtype=x.dtype)  # (F,)
    #     # (..., 3, 1) * (F,) -> (..., 3, F)
    #     xb = x.unsqueeze(-1) * freqs
    #     xb = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # (..., 3, 2F)
    #     xb = xb.view(*x.shape[:-1], -1)                         # (..., 3*2F)
    #     return torch.cat([x, xb], dim=-1)                       # (..., 3*(1+2F))

    def forward(self, ray_bundle: RayBundle):
        """
        Forward pass of NeRF MLP

        Args:
            ray_bundle: RayBundle object containing sample points

        Returns:
            dict with 'density' and 'feature' (color) keys
        """
        # Extract sample points from ray bundle
        # sample_points has shape (..., n_samples, 3)
        sample_points = ray_bundle.sample_points
        original_shape = sample_points.shape[:-1]  # Save shape for later
        sample_points_flat = sample_points.view(-1, 3)  # Flatten to (N, 3)

        # Apply positional encoding to positions
        embedded_xyz = self.harmonic_embedding_xyz(sample_points_flat)  # (N, embedding_dim_xyz)

        # Pass through MLP
        if self.append_xyz:
            # If using skip connections, pass embedded input twice
            features = self.mlp_xyz(embedded_xyz, embedded_xyz)
        else:
            features = self.mlp_xyz(embedded_xyz)

        # Get density (with ReLU to ensure non-negative)
        raw_density = self.density_layer(features)  # (N, 1)
        density = torch.nn.functional.relu(raw_density)

        # Get color (with Sigmoid to map to [0, 1])
        raw_color = self.color_layer(features)  # (N, 3)
        color = torch.sigmoid(raw_color)

        # Reshape outputs back to original batch shape
        density = density.view(*original_shape, 1)
        color = color.view(*original_shape, 3)

        return {
            'density': density.squeeze(-1),  # Remove last dimension for density
            'feature': color.view(-1, 3)  # Flatten color for compatibility
        }


class NeuralSurface(torch.nn.Module):
    """
    Neural SDF: MLP that predicts signed distance from any 3D point to the surface.

    Key differences from NeRF:
    1. Output: SDF (unbounded, can be negative) vs Density (non-negative)
    2. Activation: NO activation on SDF output vs ReLU for density
    3. Training: Supervised on point cloud + eikonal regularization
    """
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        # TODO (Q7): Implement Neural Surface MLP to output per-point color

        # ===== STEP 1: Positional Encoding =====
        # Use harmonic embedding to capture high-frequency details
        # Same as NeRF, but we only need position (no view direction for SDF)
        n_harmonic_functions = cfg.get('n_harmonic_functions_xyz', cfg.get('n_harmonic_functions', 6))
        self.harmonic_embedding = HarmonicEmbedding(
            in_channels=3,  # xyz coordinates
            n_harmonic_functions=n_harmonic_functions,
            include_input=True
        )

        embedding_dim = self.harmonic_embedding.output_dim

        # ===== STEP 2: MLP Architecture =====
        # Configuration - support both naming conventions
        n_layers = cfg.get('n_layers_distance', cfg.get('n_layers', 8))
        hidden_dim = cfg.get('n_hidden_neurons_distance', cfg.get('n_hidden_neurons', 256))
        skip_connections = cfg.get('append_distance', cfg.get('skip_connections', []))

        # Build MLP with skip connections (like NeRF)
        # Skip connections help with gradient flow and learning details
        if skip_connections:
            self.sdf_mlp = MLPWithInputSkips(
                n_layers=n_layers,
                input_dim=embedding_dim,
                output_dim=hidden_dim,
                skip_dim=embedding_dim,
                hidden_dim=hidden_dim,
                input_skips=skip_connections
            )
        else:
            # Simple MLP without skips
            layers = []
            for i in range(n_layers):
                if i == 0:
                    layers.append(torch.nn.Linear(embedding_dim, hidden_dim))
                else:
                    layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(torch.nn.ReLU(True))
            self.sdf_mlp = torch.nn.Sequential(*layers)

        # ===== STEP 3: Output Heads =====

        # Distance head: outputs SDF value (1 scalar)
        # CRITICAL: NO activation! SDF can be negative (inside) or positive (outside)
        self.distance_layer = torch.nn.Linear(hidden_dim, 1)

        # Color head: outputs RGB (3 values)
        # Can use separate MLP for color or simple head
        n_layers_color = cfg.get('n_layers_color', 2)
        n_hidden_neurons_color = cfg.get('n_hidden_neurons_color', hidden_dim)

        # Build color MLP
        color_layers = []
        for i in range(n_layers_color):
            if i == 0:
                color_layers.append(torch.nn.Linear(hidden_dim, n_hidden_neurons_color))
            else:
                color_layers.append(torch.nn.Linear(n_hidden_neurons_color, n_hidden_neurons_color))
            color_layers.append(torch.nn.ReLU(True))
        # Final color output layer
        color_layers.append(torch.nn.Linear(n_hidden_neurons_color, 3))
        color_layers.append(torch.nn.Sigmoid())  # Map to [0, 1] for RGB

        self.color_layer = torch.nn.Sequential(*color_layers)

        self.cfg = cfg
        self.skip_connections = skip_connections

    def get_distance(
        self,
        points
    ):
        '''
        Predict SDF value for input points.

        Args:
            points: (N, 3) or (..., 3) tensor of 3D coordinates

        Output:
            distance: (N, 1) tensor of signed distances
                     - Negative: inside surface
                     - Zero: on surface
                     - Positive: outside surface
        '''
        # Flatten to (N, 3)
        points = points.view(-1, 3)

        # Step 1: Positional encoding
        # Transform xyz → high-dimensional embedding
        embedded = self.harmonic_embedding(points)  # (N, embedding_dim)

        # Step 2: Pass through MLP
        if self.skip_connections:
            # With skip connections
            features = self.sdf_mlp(embedded, embedded)
        else:
            # Without skip connections
            features = self.sdf_mlp(embedded)

        # Step 3: Predict SDF
        # IMPORTANT: No activation! SDF can be any real number
        distance = self.distance_layer(features)  # (N, 1)

        return distance

    def get_color(
        self,
        points
    ):
        '''
        Predict RGB color for input points.

        Args:
            points: (N, 3) tensor of 3D coordinates

        Output:
            color: (N, 3) tensor of RGB values in [0, 1]
        '''
        points = points.view(-1, 3)

        # Reuse computation from distance prediction
        # Step 1: Positional encoding
        embedded = self.harmonic_embedding(points)

        # Step 2: MLP features
        if self.skip_connections:
            features = self.sdf_mlp(embedded, embedded)
        else:
            features = self.sdf_mlp(embedded)

        # Step 3: Predict color
        color = self.color_layer(features)  # (N, 3) in [0, 1]

        return color

    def get_distance_color(
        self,
        points
    ):
        '''
        Efficiently compute both distance and color by sharing computation.

        Output:
            distance: (N, 1) tensor
            color: (N, 3) tensor
        '''
        points = points.view(-1, 3)

        # Shared computation: embedding and MLP features
        embedded = self.harmonic_embedding(points)

        if self.skip_connections:
            features = self.sdf_mlp(embedded, embedded)
        else:
            features = self.sdf_mlp(embedded)

        # Separate heads
        distance = self.distance_layer(features)  # (N, 1)
        color = self.color_layer(features)  # (N, 3)

        return distance, color

    def forward(self, points):
        """Default forward: return SDF"""
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
