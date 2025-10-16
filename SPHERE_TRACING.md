# Sphere Tracing Implementation Guide (Q5)

## Concept Overview

**Sphere tracing** is an efficient ray marching technique for rendering implicit surfaces defined by Signed Distance Functions (SDFs).

### What is an SDF?

An SDF maps any 3D point to the **signed distance** to the nearest surface:
- **Negative**: inside the object
- **Zero**: on the surface
- **Positive**: outside the object

The absolute value `|SDF(p)|` tells you the minimum distance to the surface from point `p`.

### Why Sphere Tracing Works

The key insight: If `SDF(p) = d`, then there's a **sphere of radius d** centered at `p` that contains no surface. So you can safely march forward by distance `d` along the ray!

```
         Ray →
Origin ----d1----> p1 ----d2----> p2 --d3-> p3 (hit!)
                    ↓              ↓         ↓
                   SDF=d1        SDF=d2    SDF≈0
```

## Algorithm Breakdown

### Step 1: Initialize
```python
t = torch.ones(N, 1, device=device) * self.near  # Distance along ray
points = origins + t * directions                 # Current positions
hit_mask = torch.zeros(N, 1, device=device, dtype=torch.bool)
```

- Start at `near` plane (not origin, to avoid numerical issues)
- Track distance `t` along each ray
- `hit_mask`: False = still marching, True = hit surface

### Step 2: Sphere Tracing Loop
```python
for i in range(self.max_iters):
    # Query SDF at current positions
    sdf_values = implicit_fn(points)  # How far to surface?

    # Check if close enough to surface
    newly_hit = (torch.abs(sdf_values) < surface_threshold)
    hit_mask = hit_mask | newly_hit

    # March forward (only for active rays)
    active_mask = ~hit_mask
    t = t + sdf_values * active_mask.float()

    # Update positions
    points = origins + t * directions
```

#### Key Points:

1. **SDF Query**: `sdf_values = implicit_fn(points)`
   - Returns signed distance to nearest surface
   - Positive = outside, negative = inside

2. **Hit Detection**: `torch.abs(sdf_values) < threshold`
   - Threshold (e.g., 1e-3) determines precision
   - Smaller = more accurate but more iterations

3. **Active Mask**: Only march rays that haven't hit yet
   - Once a ray hits, it stops marching
   - Prevents overshooting the surface

4. **Distance Update**: `t = t + sdf_values`
   - March forward by the safe distance
   - Adaptive step size (large steps in empty space, small near surface)

### Step 3: Bounds Checking
```python
t = torch.clamp(t, self.near, self.far)
```

- Prevent marching beyond `far` plane
- Rays that exceed `far` have "missed" the surface

### Step 4: Early Termination
```python
if hit_mask.all():
    break
```

- If all rays have hit, no need to continue
- Saves computation

## Comparison with Naive Ray Marching

| Method | Step Size | Efficiency |
|--------|-----------|------------|
| **Naive Ray Marching** | Fixed small steps | Slow, many iterations |
| **Sphere Tracing** | Adaptive (large in empty space) | Fast, fewer iterations |

### Example:

**Naive**: 1000 steps of size 0.01 = 1000 SDF queries
**Sphere Tracing**: ~20 steps (adaptive) = 20 SDF queries ✓

## Visual Explanation

```
Scene: Torus centered at origin

Ray 1 (hits torus):
  Origin → (near) → far from torus → march large step
                  → closer to torus → march medium step
                  → very close → march tiny step
                  → |SDF| < threshold → HIT! ✓

Ray 2 (misses torus):
  Origin → (near) → march large steps through empty space
                  → exceed far plane → MISS
```

## Implementation Details

### Critical Bug in Original Code

**Original (WRONG)**:
```python
mask = torch.ones(N, 1, dtype=torch.bool)  # Start with True
for _ in range(max_iters):
    hit = (torch.abs(sdf) < threshold)
    mask = hit | mask  # Always True! Bug!
```

**Problem**: `mask` starts as all True, so `hit | mask` is always True. Wrong logic!

**Fixed (CORRECT)**:
```python
hit_mask = torch.zeros(N, 1, dtype=torch.bool)  # Start with False
for _ in range(max_iters):
    newly_hit = (torch.abs(sdf) < threshold)
    hit_mask = hit_mask | newly_hit  # Accumulate hits
```

### Active Masking

```python
active_mask = ~hit_mask  # Rays still marching
t = t + sdf_values * active_mask.float()  # Only update active
```

This ensures:
- Rays that hit stop marching (prevent overshooting)
- Rays that miss keep marching until `far` or `max_iters`

## Configuration Parameters

From `configs/torus_surface.yaml`:
```yaml
near: 0.1    # Start marching from this distance
far: 10.0    # Stop if exceed this distance (miss)
max_iters: 100  # Maximum marching iterations
```

**Tuning**:
- `surface_threshold = 1e-3`: Smaller = more accurate, more iterations
- `max_iters`: Higher = better for complex geometry, slower
- `near/far`: Define the ray marching bounds

## Running the Code

```bash
python -m surface_rendering_main --config-name=torus_surface
```

This will:
1. Create a torus SDF
2. Generate cameras around the torus
3. For each pixel, cast a ray and sphere trace
4. Render based on hit/miss
5. Save `images/part_5.gif`

## Expected Output

You should see a smooth, anti-aliased rendering of a torus rotating in 3D. The sphere tracing algorithm efficiently finds ray-surface intersections.

## Advantages of Sphere Tracing

1. **Efficient**: Adaptive step sizes (large in empty space)
2. **Accurate**: Can control precision with threshold
3. **Simple**: No complex acceleration structures needed
4. **Flexible**: Works with any SDF (primitives or neural networks)

## Common Issues & Debugging

### Issue 1: Black image
- **Cause**: All rays missing (no hits)
- **Fix**: Check `near`, `far`, scene geometry

### Issue 2: Noisy/artifacts
- **Cause**: Threshold too large
- **Fix**: Decrease `surface_threshold` (e.g., 1e-4)

### Issue 3: Slow rendering
- **Cause**: Too many iterations
- **Fix**: Increase threshold or reduce `max_iters`

### Issue 4: Rays passing through surface
- **Cause**: Not stopping rays that hit
- **Fix**: Use active mask (implemented above)

## Math Behind It

### Why is it "safe" to march by SDF distance?

**Theorem**: If `SDF(p) = d`, then all points within distance `d` of `p` are at least distance `d` from the surface.

**Proof sketch**:
- SDF is the minimum distance to surface
- A sphere of radius `d` centered at `p` contains no surface points
- Therefore, marching `≤ d` along any direction is safe

### Convergence

Sphere tracing converges **linearly** near the surface:
- Far from surface: large steps (fast)
- Near surface: small steps (accurate)
- Converges when `|SDF| < threshold`

## Extension: Neural SDFs

The beauty of sphere tracing: works with **learned** SDFs too!
- Replace analytical SDF (torus) with neural network
- Network predicts SDF at each point
- Same sphere tracing algorithm works!

This is the foundation for Q6 (Neural SDF) and Q7 (VolSDF).

## Summary

Sphere tracing is an elegant algorithm that leverages the mathematical properties of SDFs to efficiently render implicit surfaces. Key insights:

1. SDF tells you how far you can safely march
2. Adaptive step sizes (large → small as you approach surface)
3. Stop when close enough (|SDF| < threshold)
4. Track active rays to prevent overshooting

The implementation correctly handles:
- Starting from near plane
- Active masking (stop rays that hit)
- Bounds checking (far plane)
- Early termination (efficiency)
