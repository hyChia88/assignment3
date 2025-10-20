# Neural SDF Implementation Guide (Q6)

## Table of Contents
1. [Concept: SDF vs Density](#concept-sdf-vs-density)
2. [Architecture Breakdown](#architecture-breakdown)
3. [Eikonal Regularization](#eikonal-regularization)
4. [Training Strategy](#training-strategy)
5. [Implementation Details](#implementation-details)

---

## Concept: SDF vs Density

### The Fundamental Difference

| **Property** | **NeRF Density** | **Neural SDF** |
|-------------|------------------|----------------|
| **Output Range** | [0, ∞) | (-∞, ∞) |
| **Meaning** | How "solid" a point is | Distance to surface |
| **Negative values** | ❌ Not allowed | ✓ Inside surface |
| **Zero** | Empty space | **On the surface** |
| **Positive** | Solid material | Outside surface |
| **Activation** | ReLU (force ≥ 0) | **None** (unbounded) |

###Visual Comparison:

```
NeRF Density:
    density(x) ≥ 0  everywhere
    Surface: where density is high

    Empty  → Solid region → Empty
    ——————————————————————————————
    0      5    10    5      0     ← density values


Neural SDF:
    SDF(x) can be ANY real number
    Surface: where SDF = 0

    Outside → Surface → Inside
    ——————————————————————————————
    +3  +2  +1   0   -1  -2  -3    ← SDF values
```

### Key Insight

**NeRF**: Learns a "cloud" of density
**Neural SDF**: Learns a **sharp boundary** (the zero-level set)

---

## Architecture Breakdown

### Step 1: Positional Encoding (Same as NeRF)

```python
self.harmonic_embedding = HarmonicEmbedding(
    in_channels=3,           # xyz coordinates
    n_harmonic_functions=6,  # Frequency bands
    include_input=True       # Include original xyz
)
```

**Why?** MLPs struggle with high-frequency details. Positional encoding maps:
```
xyz → [xyz, sin(2⁰π·xyz), cos(2⁰π·xyz), sin(2¹π·xyz), cos(2¹π·xyz), ...]
```

**Effect**:
- Input: 3D → Output: ~39D (for 6 frequencies)
- Enables learning fine surface details

### Step 2: MLP with Skip Connections

```python
if skip_connections:
    self.sdf_mlp = MLPWithInputSkips(
        n_layers=8,
        input_dim=embedding_dim,
        hidden_dim=256,
        skip_dim=embedding_dim,
        input_skips=[4]  # Add skip at layer 4
    )
```

**Architecture**:
```
Input (39D)
    ↓
Layer 0: Linear(39, 256) + ReLU
    ↓
Layer 1-3: Linear(256, 256) + ReLU
    ↓
Layer 4: Concat([features, input]) → Linear(256+39, 256) + ReLU  ← SKIP!
    ↓
Layer 5-7: Linear(256, 256) + ReLU
    ↓
Features (256D)
```

**Why skip connections?**
- Helps gradients flow during backprop
- Allows network to learn residuals (fine details)
- Prevents vanishing gradients in deep networks

### Step 3: Output Head (Critical Difference!)

```python
# NeRF (Density):
self.density_layer = Linear(256, 1)
density = ReLU(raw_output)  # ← Force non-negative!

# Neural SDF:
self.distance_layer = Linear(256, 1)
distance = raw_output  # ← NO activation! Can be any value!
```

**Why no activation for SDF?**
```python
# Example points around a sphere
inside_point  = [0, 0, 0]    # SDF should be negative
surface_point = [1, 0, 0]    # SDF should be 0
outside_point = [2, 0, 0]    # SDF should be positive

# With ReLU (WRONG):
SDF([0,0,0]) = ReLU(-1) = 0  ❌ Should be negative!

# Without activation (CORRECT):
SDF([0,0,0]) = -1  ✓
SDF([1,0,0]) = 0   ✓
SDF([2,0,0]) = +1  ✓
```

---

## Eikonal Regularization

### The Problem: Arbitrary Functions

Without constraints, the MLP could learn ANY function that happens to be zero at training points:

```python
# Bad example (not a distance function):
SDF(training_point) = 0      ✓ Matches training
SDF(nearby_point)   = 1000   ❌ Not a valid distance!
SDF(far_point)      = -500   ❌ Makes no sense!
```

### The Solution: Eikonal Equation

A **valid** SDF must satisfy:
```
||∇SDF(x)|| = 1  for all x
```

**Meaning**: The gradient (direction of steepest change) has unit length everywhere.

### Why This Works

**Geometric interpretation**:
- ∇SDF points toward the nearest surface point
- Its magnitude equals the rate of distance change
- For a true distance function, moving 1 unit changes distance by 1
- Therefore: ||∇SDF|| = 1

**Visual**:
```
        Surface (SDF=0)
            ↓
    +2  +1  0  -1  -2
     ↓   ↓  ↓   ↓   ↓
    Each arrow has length 1 (unit norm)
```

### Implementation

```python
def eikonal_loss(gradients):
    """
    Args:
        gradients: (N, 3) tensor of ∇SDF

    Returns:
        loss: mean((||∇SDF|| - 1)²)
    """
    gradient_norm = torch.linalg.norm(gradients, dim=-1)  # ||∇SDF||
    loss = torch.mean((gradient_norm - 1.0) ** 2)
    return loss
```

**How gradients are computed**:
```python
# In NeuralSurface.get_distance_and_gradient():
points = points.requires_grad_(True)  # Enable gradient tracking
distance = self.get_distance(points)   # Forward pass

# Compute ∇SDF w.r.t. input points
gradient = autograd.grad(
    distance,    # Output
    points,      # Input (w.r.t. which we compute gradient)
    ...
)[0]  # → (N, 3) tensor of ∇SDF
```

---

## Training Strategy

### Loss Function

```python
# Total loss = Data loss + Regularization
total_loss = sdf_loss + λ_eikonal * eikonal_loss

# Component 1: Data loss (supervision from point cloud)
sdf_loss = mean(SDF(surface_points)²)
# → Forces SDF=0 at known surface points

# Component 2: Eikonal regularization
eikonal_loss = mean((||∇SDF|| - 1)²)
# → Ensures valid distance function everywhere
```

### Why Both Losses?

**Data loss alone** (without eikonal):
- ❌ Network might learn: SDF=0 at training points, chaos elsewhere
- ❌ No guarantee it's a distance function

**Eikonal alone** (without data loss):
- ❌ Many functions satisfy ||∇f|| = 1 (e.g., planes)
- ❌ Need data to anchor the surface

**Both together**:
- ✓ SDF=0 at training points (data loss)
- ✓ Valid distance function everywhere (eikonal)
- ✓ Generalizes to unseen points!

### Typical Hyperparameters

```python
λ_eikonal = 0.1  # Weight for eikonal regularization

# For each training iteration:
# 1. Sample points from point cloud
# 2. Sample random points in space
# 3. Compute losses
# 4. total_loss = sdf_loss + 0.1 * eikonal_loss
# 5. Backprop
```

---

## Implementation Details

### Full Forward Pass

```python
def get_distance(self, points):
    # points: (N, 3)

    # Step 1: Positional encoding
    embedded = self.harmonic_embedding(points)  # (N, 39)

    # Step 2: MLP processing
    if self.skip_connections:
        features = self.sdf_mlp(embedded, embedded)  # (N, 256)
    else:
        features = self.sdf_mlp(embedded)

    # Step 3: Output SDF
    distance = self.distance_layer(features)  # (N, 1)
    # NO activation! Can be any real number

    return distance
```

### Computing Gradients for Eikonal Loss

```python
def get_distance_and_gradient(self, points):
    with torch.enable_grad():
        points = points.requires_grad_(True)  # Track gradients
        distance = self.get_distance(points)   # Forward

        # Backprop to get ∇SDF w.r.t. input points
        gradient = autograd.grad(
            outputs=distance,
            inputs=points,
            grad_outputs=torch.ones_like(distance),
            create_graph=True,  # Allow higher-order derivatives
            retain_graph=True,
            only_inputs=True
        )[0]  # (N, 3)

    return distance, gradient
```

### Training Loop Pseudocode

```python
for epoch in range(num_epochs):
    # Sample points from point cloud (surface points)
    surface_points = sample_from_point_cloud(batch_size)

    # Sample random points in space
    random_points = sample_random_in_bbox(batch_size)

    # Compute SDF and gradients
    sdf_surface, grad_surface = model.get_distance_and_gradient(surface_points)
    sdf_random, grad_random = model.get_distance_and_gradient(random_points)

    # Data loss: SDF should be 0 at surface points
    data_loss = torch.mean(sdf_surface ** 2)

    # Eikonal loss: ||∇SDF|| should be 1 everywhere
    eikonal_loss_surface = eikonal_loss(grad_surface)
    eikonal_loss_random = eikonal_loss(grad_random)
    eik_loss = eikonal_loss_surface + eikonal_loss_random

    # Total loss
    loss = data_loss + λ_eikonal * eik_loss

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Comparison Summary

### NeRF (Density Field)
- **Learn**: Volume density at every point
- **Output**: density ∈ [0, ∞)
- **Activation**: ReLU
- **Surface**: Implicit (high density region)
- **Training**: MSE on RGB colors
- **Regularization**: None required

### Neural SDF
- **Learn**: Distance to surface
- **Output**: distance ∈ (-∞, ∞)
- **Activation**: **None**
- **Surface**: Explicit (SDF = 0)
- **Training**: MSE on point cloud
- **Regularization**: **Eikonal** (||∇SDF|| = 1)

---

## Key Takeaways

1. **No activation on SDF output**: Must allow negative values for "inside"
2. **Eikonal regularization is essential**: Ensures valid distance function
3. **Gradients computed via autograd**: ∇SDF w.r.t. input coordinates
4. **Skip connections help**: Deep networks need them for gradient flow
5. **Positional encoding**: Same as NeRF, enables high-frequency details

---

## Next Steps (Q7: VolSDF)

After implementing Neural SDF, you can:
1. Add color prediction (another MLP head)
2. Convert SDF → density for volume rendering
3. Combine benefits of both: surface representation + volume rendering!

This is what VolSDF does in Q7.