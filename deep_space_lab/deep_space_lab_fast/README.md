
# Deep-Space Lab (FAST Edition)
- 全流程：Halo 族线 + STM 真导数 + 多段/配点修正 + SRP/J2 漂移 + NSGA-II 三目标 + OD 残差/2σ。
- 仅依赖：numpy、matplotlib。默认参数均为“**快速出图**”，1~2 分钟内可完成。
- 入口：`python run_all_fast.py`，或选择 `halo | pareto | od` 子集：`python run_all_fast.py halo`

## 输出
生成的图在 `deep_space_lab_fast/outputs/`。总计十余张图（族线、谱半径、迭代/耗时、SRP/J2、Pareto、超体积、ΔV组成、OD残差、2σ）。
