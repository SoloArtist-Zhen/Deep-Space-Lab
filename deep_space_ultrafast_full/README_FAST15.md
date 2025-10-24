# Deep-Space Ultra-Fast (15 Figures)
Self-contained (numpy + matplotlib). Produces **15 advanced figures** (no bars, no plain line plots):
- Zero-velocity contours; vector field streamplot
- 3D halo-like trajectory (colored by speed); Poincaré section
- Spectral radius polar sketch; SRP/J2 drift heatmap
- Solver paths (quiver); Collocation defect contour
- 3D Pareto front; Polar ΔV heatmap
- Hypervolume proxy surface; Residual spectrogram
- Filled covariance ellipse; CRB heatmap
- Monodromy eigenvalues on complex plane

Run:
```bash
python -u run_fast15.py
```
Outputs go to `./fast15_outputs/`.
