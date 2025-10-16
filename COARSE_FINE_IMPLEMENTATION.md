# NeRF Coarse/Fine Sampling Implementation (Q4.2)

## Overview

This implementation adds hierarchical volumetric sampling to NeRF, following the original paper. The strategy uses two networks (coarse and fine) with importance sampling to concentrate samples in regions likely to contain visible content.

## Implementation Details

### 1. Hierarchical Sampler (`sampler.py`)

**Class: `HierarchicalSampler`**

- Performs stratified sampling initially (like `StratifiedRaysampler`)
- When given coarse weights, performs importance sampling using `sample_pdf()`:
  - Converts weights to a probability density function (PDF)
  - Computes cumulative distribution function (CDF)
  - Inverts CDF using `torch.searchsorted()` to sample more points where density is high
  - Combines coarse and fine samples and sorts them

**Key Method: `sample_pdf()`**
- Implements inverse transform sampling
- Samples more points in regions with high coarse density
- Uses linear interpolation between bin edges

### 2. Coarse-Fine Model (`volume_rendering_main.py`)

**Class: `CoarseFineModel`**

Architecture:
- **Coarse network**: Uses fewer samples (64 in default config)
- **Fine network**: Uses hierarchical sampling (128 additional samples)
- Both networks have the same MLP architecture but separate parameters

**Forward Pass (Two-Pass Rendering)**:
1. **Coarse Pass**:
   - Sample points uniformly along rays
   - Render with coarse network
   - Extract weights (importance) from rendering equation

2. **Fine Pass**:
   - Use coarse weights for importance sampling
   - Sample additional points where density is high
   - Combine with coarse samples and sort
   - Render with fine network using all samples

### 3. Renderer Updates (`renderer.py`)

**Method: `_render_with_implicit()`**
- Refactored rendering logic into a helper method
- Returns weights in addition to color and depth
- Weights are used by hierarchical sampler for importance sampling

### 4. Training Function

**Function: `train_nerf_coarse_fine()`**
- Trains both networks simultaneously
- Computes losses for both coarse and fine outputs
- Total loss = `coarse_weight * loss_coarse + fine_weight * loss_fine`
- Default weights: coarse=0.5, fine=1.0 (fine network weighted more)

### 5. Configuration (`configs/nerf_lego_coarse_fine.yaml`)

Key parameters:
- `sampler_coarse`: 64 stratified samples
- `sampler_fine`: 128 hierarchical samples
- Total samples per ray: 64 + 128 = 192 (sorted and combined)
- Larger network: 8 layers, 256 hidden neurons, higher positional encoding

## Usage

```bash
python volume_rendering_main.py --config-name=nerf_lego_coarse_fine
```

The training will:
- Print separate losses for coarse and fine networks
- Render test images periodically (saved as `images/part_4_2_epoch_*.gif`)
- Use the fine network for final rendering

## Trade-offs Discussion

### Speed vs Quality

**Advantages**:
1. **Better quality**: Fine samples concentrated where they matter most
2. **More efficient**: Fewer wasted samples in empty space
3. **Better geometry**: Coarse network provides geometry prior for fine network
4. **Reduced aliasing**: Denser sampling near surfaces

**Disadvantages**:
1. **Slower training**: Two forward passes per iteration (coarse + fine)
2. **More memory**: Two full networks with separate parameters
3. **Complexity**: More hyperparameters to tune (sample counts, loss weights)
4. **Inference cost**: Need both networks for training (can use only fine for inference)

### Performance Metrics

**Training time**: ~2x slower than single network (two forward passes)
**Memory usage**: ~2x model parameters (separate coarse/fine networks)
**Sample efficiency**: Better - 192 samples with importance > 256 uniform samples
**Quality**: Significantly better, especially for fine details and sharp edges

### When to Use Coarse/Fine

**Use when**:
- Quality is paramount
- Training time is not a constraint
- Scenes have complex geometry with empty space
- Need sharp, detailed results

**Skip when**:
- Fast iteration is needed
- Limited GPU memory
- Simple scenes with uniform density
- Real-time requirements

## Technical Notes

### Importance Sampling Math

Given coarse weights `w_i` for samples along a ray:

1. Normalize to PDF: `pdf_i = w_i / Σw_i`
2. Compute CDF: `cdf_i = Σ(pdf_0...pdf_i)`
3. Sample uniform values: `u ~ Uniform(0, 1)`
4. Invert CDF: find `i` where `cdf_{i-1} < u < cdf_i`
5. Interpolate within bin for continuous sampling

### Why This Works

- Regions with high density (high weights) get more samples
- Empty regions get fewer samples
- Fine network sees both coarse and fine samples, learns details
- Coarse network acts as a regularizer, preventing overfitting

## Files Modified

1. `sampler.py`: Added `HierarchicalSampler` class
2. `renderer.py`: Added `_render_with_implicit()` helper method
3. `volume_rendering_main.py`: Added `CoarseFineModel` and `train_nerf_coarse_fine()`
4. `configs/nerf_lego_coarse_fine.yaml`: New configuration file

## Future Improvements

- **Adaptive sampling**: Dynamically adjust number of fine samples
- **Single network**: Share early layers between coarse/fine
- **Progressive training**: Start with coarse only, add fine later
- **MipNeRF integration**: Combine with anti-aliased sampling