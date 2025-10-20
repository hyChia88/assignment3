# Assignment 3 Submission

## A. Neural Volume Rendering

### 1. Differentiable Volume Rendering

**Pixel grid visualization**  
![Grid](images/grid.png)

**Ray visualization**  
![Rays](images/rays.png)

**Stratified point samples**  
![Sample Points](images/sample_points.png)

**Spiral rendering**   
![Part 1](images/part_1.gif)

**Depth map**  
![Depth](images/depth.png)

### 2. Optimizing a Basic Implicit Volume

**Optimized box volume**  
![Part 2](images/part_2.gif)

### 3. Optimizing a Neural Radiance Field (NeRF)

**NeRF rendering of lego bulldozer**  
![Part 3](images/part_3.gif)

### 4.
Run with `python volume_rendering_main.py --config-name=nerf_lego type=train_nerf`
config:
```
epochs=30
```

```
Saved nerf rendering to images/part_3.gif
Epoch: 0021, Loss: 0.007249: : 100it [00:14,  7.13it/s]
Epoch: 0022, Loss: 0.007989: : 100it [00:11,  8.36it/s]
Epoch: 0023, Loss: 0.006072: : 100it [00:12,  8.21it/s]
Epoch: 0024, Loss: 0.008253: : 100it [00:13,  7.28it/s]
Epoch: 0025, Loss: 0.007004: : 100it [00:11,  8.41it/s]
Epoch: 0026, Loss: 0.006192: : 100it [00:12,  8.09it/s]
Epoch: 0027, Loss: 0.009732: : 100it [00:13,  7.30it/s]
Epoch: 0028, Loss: 0.006877: : 100it [00:12,  8.10it/s]
Epoch: 0029, Loss: 0.005190: : 100it [00:14,  6.87it/s]
```
---
## B. Neural Surface Rendering

### 5. Sphere Tracing

**Torus rendering**  
![Part 5](images/part_5.gif)

### 6. Optimizing a Neural SDF

**Input point cloud**  
![Part 6 Input](images/part_6_input.gif)

### 7. VolSDF

**Learned SDF geometry**  
![Part 7 Geometry](images/part_7_geometry.gif)

**VolSDF rendering with color**  
![Part 7](images/part_7.gif)