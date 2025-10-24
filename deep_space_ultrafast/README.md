
# Deep-Space Ultra-Fast Suite (No Bar/Line Charts + Algo Compare)
目标：和 smoke 一样的速度（秒级），但覆盖深空 4 个核心要素，并**只输出散点/云图/3D 图**，不出柱状图/折线图。
生成 4 张图：
1) `crtbp_orbit_scatter.png` —— 2D CRTBP 小轨道点云（以时间上色）。
2) `srp_j2_overlay_scatter.png` —— 开/关 SRP+J2 的轨道散点叠图。
3) `pareto_front_3d.png` —— 迷你 NSGA-II 三目标前沿（ΔV–TOF–Miss）3D 散点。
4) `alg_compare_scatter.png` —— **单段 shooting vs. 简化 collocation** 在小型 BVP 上的“残差 vs 耗时”散点对比。

依赖：`numpy`, `matplotlib`（无需其它库）。
运行：
```bash
python -u run_ultrafast.py
```
所有图片保存在 `./outputs/`，算法对比与优化摘要保存在 `./outputs/summary.json`。
