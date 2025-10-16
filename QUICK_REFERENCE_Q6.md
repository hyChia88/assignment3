# Quick Reference: Neural SDF vs NeRF

## Critical Differences at a Glance

### 1. Output Layer

```python
# ❌ WRONG (copying NeRF):
distance = torch.nn.functional.relu(self.distance_layer(features))

# ✓ CORRECT (Neural SDF):
distance = self.distance_layer(features)  # NO activation!
```

**Why?** SDF must be **negative inside**, zero on surface, positive outside.

---

### 2. Loss Function

```python
# NeRF:
loss = MSE(predicted_color, ground_truth_color)

# Neural SDF:
data_loss = MSE(SDF(surface_points), 0)  # SDF=0 at surface
eikonal_loss = MSE(||∇SDF||, 1)          # Unit gradient norm
total_loss = data_loss + λ * eikonal_loss
```

---

### 3. The Eikonal Equation

```python
def eikonal_loss(gradients):
    # gradients: (N, 3) - computed via autograd
    gradient_norm = torch.linalg.norm(gradients, dim=-1)  # (N,)
    return torch.mean((gradient_norm - 1.0) ** 2)
```

**Physical meaning**: For a true distance function, moving 1 unit changes distance by 1.

---

### 4. Architecture Components

| Component | NeRF | Neural SDF |
|-----------|------|------------|
| Input encoding | ✓ Positional encoding | ✓ Positional encoding |
| MLP depth | 8 layers | 8 layers |
| Skip connections | ✓ At layer 4 | ✓ At layer 4 |
| Hidden dim | 256 | 256 |
| Output activation | **Sigmoid** (color), **ReLU** (density) | **None** (SDF) |
| View dependence | Yes (for color) | No (SDF is geometric) |

---

## Common Mistakes

### Mistake 1: Adding ReLU
```python
# ❌ This breaks negative SDF values!
distance = F.relu(self.distance_layer(features))
```

### Mistake 2: Forgetting Eikonal
```python
# ❌ Network learns arbitrary function, not a distance field
loss = MSE(SDF(points), 0)  # Missing eikonal regularization!
```

### Mistake 3: Wrong Gradient Computation
```python
# ❌ Computing gradient w.r.t. MLP weights (not what we want!)
gradient = autograd.grad(distance, self.parameters())

# ✓ Compute gradient w.r.t. INPUT POINTS
points = points.requires_grad_(True)
distance = self.get_distance(points)
gradient = autograd.grad(distance, points, ...)  # ∇SDF w.r.t. xyz
```

---

## Checklist Before Running

- [ ] No activation on SDF output
- [ ] Positional encoding implemented
- [ ] Skip connections at layer 4
- [ ] Eikonal loss implemented correctly
- [ ] Gradients computed w.r.t. input points, not weights
- [ ] Both data loss and eikonal loss in training

---

## Debugging Tips

### Issue: SDF values all positive
**Cause**: ReLU activation on output
**Fix**: Remove activation

### Issue: Noisy/chaotic surface
**Cause**: No eikonal regularization
**Fix**: Add eikonal loss with λ=0.1

### Issue: SDF doesn't match point cloud
**Cause**: Eikonal weight too high
**Fix**: Reduce λ_eikonal (try 0.01-0.1)

### Issue: Gradient computation error
**Cause**: `create_graph=False` when training
**Fix**: Set `create_graph=True` for eikonal loss

---

## Running the Code

```bash
# Test your implementation
python -m surface_rendering_main --config-name=points_surface

# Expected output:
# - images/part_6_input.gif  (input point cloud)
# - images/part_6.gif        (learned surface)
```

---

## Key Formulas

**Data Loss**: $L_{data} = \frac{1}{N}\sum_{i=1}^N [\text{SDF}(p_i)]^2$ where $p_i$ are surface points

**Eikonal Loss**: $L_{eikonal} = \frac{1}{N}\sum_{i=1}^N [||\nabla \text{SDF}(x_i)|| - 1]^2$

**Total Loss**: $L = L_{data} + \lambda L_{eikonal}$

where $\lambda \in [0.01, 0.1]$ typically.