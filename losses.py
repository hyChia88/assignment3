import torch
import torch.nn.functional as F

def eikonal_loss(gradients):
    # TODO (Q6): Implement eikonal loss
    """
    Eikonal regularization loss.

    For a valid SDF, the gradient (∇SDF) should have unit norm everywhere:
    ||∇SDF(x)|| = 1 for all x

    This is a fundamental property of distance functions: the gradient points
    toward the nearest surface point with magnitude 1.

    Args:
        gradients: (N, 3) tensor of gradients ∇SDF for N points
                  These are computed via autograd.grad() in get_distance_and_gradient()

    Returns:
        loss: scalar tensor, mean squared deviation from unit norm
              loss = mean((||∇SDF|| - 1)²)

    Mathematical intuition:
    - If ||∇SDF|| = 1: perfect distance function
    - If ||∇SDF|| > 1: distance changes too quickly (not a true distance)
    - If ||∇SDF|| < 1: distance changes too slowly (not a true distance)
    """
    # Compute L2 norm of gradients: ||∇SDF|| for each point
    # gradients shape: (N, 3) → gradient_norm shape: (N,)
    gradient_norm = torch.linalg.norm(gradients, dim=-1)  # (N,)

    # Eikonal constraint: ||∇SDF|| should be 1
    # Compute squared deviation from 1
    loss = torch.mean((gradient_norm - 1.0) ** 2)

    return loss

def sphere_loss(signed_distance, points, radius=1.0):
    return torch.square(signed_distance[..., 0] - (torch.norm(points, dim=-1) - radius)).mean()

def get_random_points(num_points, bounds, device):
    min_bound = torch.tensor(bounds[0], device=device).unsqueeze(0)
    max_bound = torch.tensor(bounds[1], device=device).unsqueeze(0)

    return torch.rand((num_points, 3), device=device) * (max_bound - min_bound) + min_bound

def select_random_points(points, n_points):
    points_sub = points[torch.randperm(points.shape[0])]
    return points_sub.reshape(-1, 3)[:n_points]
