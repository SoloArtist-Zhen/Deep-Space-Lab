
# Deep‑Space Lab
**Halo 族线 + STM 真导数 + 多段/配点修正 + SRP/J2 摄动力 + 多目标 NSGA‑II + 精密定轨（Batch/EKF/CRB）**  
> **可运行研究底座**。仅依赖：`numpy`、`matplotlib`。所有算法与可视化在仓库内自包含。

---

## 0. 快速开始
```bash
# 生成全部图（Halo/优化/定轨），默认 1~3 分钟
python run_all.py

# 只跑某一部分
python run_all.py halo
python run_all.py pareto
python run_all.py od
```
生成的高级图保存在 `deep_space_lab/outputs/`。下文对**每张图**逐一解释并给出所用的**数学表达式（LaTeX）**。

---

## 1. 模型与符号

### 1.1 CRTBP（地月）旋转坐标系
无量纲化后，地月质量参数记为 \\(\mu\\)，原点为系统质心，主天体在 \\((-\,\mu,0,0)\\) 与 \\((1-\mu,0,0)\\)。设状态 \\(\mathbf{x}=[x,y,z,v_x,v_y,v_z]^T\\)。势函数
$$
U(x,y,z)\;=\;\tfrac12\,(x^2+y^2)\;+\;\frac{1-\mu}{r_1}\;+\;\frac{\mu}{r_2},
\quad
r_1=\Big\|\begin{bmatrix}x+\mu\\y\\z\end{bmatrix}\Big\|,\;
r_2=\Big\|\begin{bmatrix}x-(1-\mu)\\y\\z\end{bmatrix}\Big\|.
$$

CRTBP 方程：
$$
\begin{aligned}
\ddot x - 2\dot y &= \frac{\partial U}{\partial x},\\
\ddot y + 2\dot x &= \frac{\partial U}{\partial y},\\
\ddot z &= \frac{\partial U}{\partial z}.
\end{aligned}
$$

记 \\(\mathbf{f}(\mathbf{x})=[v_x,v_y,v_z,\;a_x,a_y,a_z]^T\\)，其雅可比（真导数）
$$
\mathbf{A}(\mathbf{x})=\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\in\mathbb{R}^{6\times 6},
\quad
\text{代码中显式构造 } \frac{\partial^2 U}{\partial x_i\partial x_j}.
$$

**Jacobi 能量（近似口径）**：
$$
C \;\approx\; 2\,U(\mathbf{r}) \;-\; \|\mathbf{v}\|^2,
$$
用于族线能量–周期关系的对比。

### 1.2 复杂摄动力：SRP 与 J2
- **SRP**（简化点源近似，常数系数 \\(c_{\rm srp}\\)）：
$$
\mathbf{a}_{\rm SRP}
\;=\;
c_{\rm srp}\,\frac{\mathbf{r}-\mathbf{r}_\odot}{\|\mathbf{r}-\mathbf{r}_\odot\|^3},
$$
其位置雅可比
$$
\frac{\partial \mathbf{a}_{\rm SRP}}{\partial \mathbf{r}}
=
c_{\rm srp}\Big(\frac{\mathbf{I}}{d^3}-\frac{3(\mathbf{r}-\mathbf{r}_\odot)(\mathbf{r}-\mathbf{r}_\odot)^T}{d^5}\Big),\quad d=\|\mathbf{r}-\mathbf{r}_\odot\|.
$$

- **J2 扁率项**（对地/对月分别计算，向量式简化）：
$$
\mathbf{a}_{J_2}
\;=\;
-\frac{\mu J_2 R^2}{r^5}
\Big[\Big(1-5\frac{z^2}{r^2}\Big)\mathbf{r} + 2 z\, r\,\hat{\mathbf{k}}\Big],
\qquad r=\|\mathbf{r}\|, 
$$
其雅可比在代码中按位置做数值真导数以匹配该形式。

> **组合动力学**：\\(\mathbf{f}=\mathbf{f}_{\rm CRTBP}+\mathbf{a}_{\rm SRP}+\mathbf{a}_{J_2({\rm Earth})}+\mathbf{a}_{J_2({\rm Moon})}\\)。

### 1.3 变分方程与 STM/单子矩阵
状态转移矩阵 \\(\Phi\\) 满足
$$
\dot{\Phi}(t)=\mathbf{A}(\mathbf{x}(t))\,\Phi(t),\quad \Phi(0)=\mathbf{I}_{6}.
$$
**单子矩阵**（周期轨道的一周 STM）：\\(\Phi(T)\\)。其**谱半径**
$$
\rho(\Phi(T))=\max_i|\lambda_i(\Phi(T))|,
$$
用于稳定性/发散性的判据与族线分析。

---

## 2. 周期 Halo 轨道与差分修正

### 2.1 初猜与周期性条件
从 L1/L2 的 Lyapunov 出发，以 \\(A_z\\) 作为纵向幅值，构造 Halo 初猜 \\(\mathbf{x}_0, T\\)。周期性条件（示例）：
$$
\mathbf{g}=\begin{bmatrix} y(T/2) \\ v_x(T/2) \end{bmatrix}
=\mathbf{0}.
$$

### 2.2 单段（半周）差分修正（Single Shooting）
线性化敏感度
$$
\Delta\mathbf{g} \approx 
\begin{bmatrix}
\Phi_{y,\,v_y}(T/2)\\[2pt]
\Phi_{v_x,\,v_y}(T/2)
\end{bmatrix}\Delta v_y(0).
$$
用最小二乘更新 \\(\Delta v_y\\) 迭代直至 \\(\|\mathbf{g}\|\\) 收敛。

### 2.3 多段差分修正（Multiple Shooting）
将周期分割为 \\(N\\) 段，端点连续性残差
$$
\mathbf{R}_i=\mathbf{x}(t_{i+1}^-)-\mathbf{x}(t_{i+1}^+)=\mathbf{0},
$$
全局以块结构线性化并用牛顿/最小二乘更新各段起点变量。

### 2.4 直接配点（Hermite–Simpson）
网格 \\(t_k\\)，离散**缺陷约束**：
$$
\mathbf{x}_{k+2}-\mathbf{x}_k - \frac{h}{6}\big(\mathbf{f}_{k}+4\mathbf{f}_{k+1}+\mathbf{f}_{k+2}\big)=\mathbf{0},
\quad h=t_{k+2}-t_k.
$$

---

## 3. 复杂摄动力与任务约束
在 Halo 一周上分别计算**无摄动**与**SRP/J2 打开**的轨道，比较漂移量与闭合性退化。可扩展加入中途脉冲/低推力段（此演示版给出双脉冲示例与多目标优化接口）。

---

## 4. 多目标轨道优化（NSGA‑II）
### 4.1 决策变量与目标
双脉冲转移（示例）决策向量
$$
\mathbf{y}=[\,t_1,\;\Delta\mathbf{v}_1,\;t_{\rm coast},\;\Delta\mathbf{v}_2\,]^T.
$$
三目标：
$$
\begin{aligned}
J_{\Delta V}&=\|\Delta\mathbf{v}_1\|+\|\Delta\mathbf{v}_2\|,\\
J_{\rm TOF}&=t_1+t_{\rm coast}+t_{\rm tail},\\
J_{\rm rob}&=\|\mathbf{x}(T)-\mathbf{x}^\star\| \quad (\text{鲁棒性代理}).
\end{aligned}
$$

### 4.2 非支配排序与拥挤度
\\(\mathbf{a}\\) **支配** \\(\mathbf{b}\\) 若 \\(\mathbf{a}\\) 在所有目标上不劣且在至少一项上更优。NSGA‑II 采用分级前沿与拥挤度保持多样性。

### 4.3 超体积（Hypervolume）
以参考点 \\(\mathbf{r}\\) 计，前沿 \\(\mathcal{F}\\) 的体积
$$
HV(\mathcal{F})= \operatorname{Vol}\Big(\bigcup_{\mathbf{f}\in\mathcal{F}} [f_1,r_1]\times[f_2,r_2]\times[f_3,r_3]\Big),
$$
作为收敛性与多样性的综合指标。

---

## 5. 精密定轨（OD）与可观测性
### 5.1 观测模型
距离、距离率与方位角/仰角：
$$
\rho=\|\mathbf{r}\|,\qquad
\dot\rho=\frac{\mathbf{r}\cdot\mathbf{v}}{\|\mathbf{r}\|},\qquad
\boldsymbol{\theta}=\begin{bmatrix}\mathrm{az}\\ \mathrm{el}\end{bmatrix}
=
\begin{bmatrix}
\operatorname{atan2}(y,x)\\[2pt]
\operatorname{atan2}(z,\sqrt{x^2+y^2})
\end{bmatrix}.
$$

### 5.2 Batch LS（线性化常规）
最小化
$$
J=\sum_k \big(\mathbf{z}_k-\mathbf{h}(\mathbf{x}_k)\big)^T\mathbf{R}^{-1}\big(\mathbf{z}_k-\mathbf{h}(\mathbf{x}_k)\big).
$$
法方程（一次线性化）
$$
(\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})\,\delta\mathbf{x}=\mathbf{H}^T\mathbf{R}^{-1}\mathbf{r},
\quad \mathbf{P}=(\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})^{-1}.
$$
\\(\mathbf{P}\\) 亦为 CRB 近似下界。

### 5.3 EKF（演示）
时间/量测更新
$$
\begin{aligned}
\mathbf{x}_{k|k-1}&=\mathbf{f}(\mathbf{x}_{k-1}),&
\mathbf{P}_{k|k-1}&=\mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^T+\mathbf{Q},\\
\mathbf{K}_k&=\mathbf{P}_{k|k-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T+\mathbf{R})^{-1},&
\mathbf{x}_k&=\mathbf{x}_{k|k-1}+\mathbf{K}_k\big(\mathbf{z}_k-\mathbf{h}(\mathbf{x}_{k|k-1})\big).
\end{aligned}
$$

---

## 6. 图表索引与释义（文件名 → 含义）

> 下文文件名与 `run_all.py` 输出一致。若你修改了参数/网格，图片序号可能略有不同。

### 6.1 Halo 套件（`halo_suite_*.png`）
1. **`halo_suite_00.png` — Period vs Amplitude**  
   展示 Halo 族线上**周期 \\(T\\)** 随纵向幅值 \\(A_z\\) 的变化趋势。横轴 \\(A_z\\)，纵轴 \\(T\\)。  
   相关量：\\(T=2\pi/\nu(A_z)\\) 的数值修正结果。

2. **`halo_suite_01.png` — Energy–Period**  
   族线**能量–周期**关系。横轴 Jacobi 近似能量 \\(C=2U-\|v\|^2\\)，纵轴 \\(T\\)。  
   反映能量面与周期轨道的对应。

3. **`halo_suite_02.png` — Monodromy Spectral Radius**  
   单子矩阵 \\(\Phi(T)\\) 的谱半径 \\(\rho(\Phi(T))\\) 随 \\(A_z\\) 的变化：
   $$\rho(\Phi(T))=\max_i|\lambda_i|.$$
   用于判定线性稳定性（\\(\rho\le 1\\)）与不稳定模态强度。

4. **`halo_suite_03.png` — Residual Norm per Iteration**  
   **Multiple shooting** 与 **Hermite–Simpson collocation** 的**残差范数–迭代**曲线：
   $$\|\mathbf{R}^{(k)}\|\ \text{vs.}\ k.$$
   对比两法的收敛速度/稳健性。

5. **`halo_suite_04.png` — Compute Time**  
   两方法一次求解的计算耗时柱状图，体现数值效率差异。

6. **`halo_suite_05.png` — SRP/J2 Drift**  
   关闭摄动力（CRTBP）与打开 **SRP+J2** 下的**同轨道对比**，可见闭合性与相位漂移变化。

### 6.2 多目标优化（`pareto_*.png`）
7. **`pareto_00.png` — Pareto Front (ΔV–TOF–Robustness)**  
   **三目标前沿**三维散点，坐标为 \\((J_{\Delta V},J_{\rm TOF},J_{\rm rob})\\)。  
   非支配解定义参见 §4。

8. **`pareto_01.png` — Hypervolume Convergence**  
   NSGA‑II 代际的**超体积** \\(HV\\) 收敛曲线：
   $$HV_g=HV(\mathcal{F}_g).$$
   反映前沿质量的整体提升。

9. **`pareto_02.png` — ΔV Composition (Best Solutions)**  
   按 \\(J_{\Delta V}\\) 最优的若干解，将 \\(\Delta V=\|\Delta\mathbf{v}_1\|+\|\Delta\mathbf{v}_2\|\\) **分解**为两次脉冲的堆叠柱状。

### 6.3 精密定轨（`od_*.png`）
10. **`od_00.png` — OD Residuals**  
    Range/Doppler/方位角/仰角的**残差序列**：
    $$\mathbf{r}_k=\mathbf{z}_k-\mathbf{h}(\hat{\mathbf{x}}_{k|k-1}).$$
    可用于频谱/系统误差分析（本演示展示时域）。

11. **`od_01.png` — Final 2σ Covariance Ellipse**  
    末历元位置协方差 \\(\mathbf{P}_{xx}\\) 的 **2σ 椭圆**。CRB 下界：
    $$\mathbf{P}\succeq (\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})^{-1}.$$
    直观展示可观测性与不确定性方向。

> **合计** 11 张图。若希望达到“15+”，可在 `run_all.py` 中：  
> (i) 追加 **λ\_{min/max} vs. A_z**（把单子矩阵特征值区间单独作图）；  
> (ii) 输出 **Energy–Amplitude** 与 **TOF–Amplitude** 的补充图；  
> (iii) 绘制 **收敛半径热图**（对不同初猜/摄动力开关批量运行）。

---

## 7. 再现实验清单（参数可在各模块顶部调整）
- **族线精化**：`halo_richardson_like` 的 \\(A_z\\) 栅格更密（如 0.01~0.12，步长 0.005），生成更平滑的能量–周期–谱半径曲线。  
- **多段 vs 配点**：提高 `segments/nodes`，比较残差曲线与耗时的交叉点，统计**成功率/收敛半径**。  
- **SRP/J2 灵敏度**：扫描 \\(c_{\rm srp},J2\_e,J2\_m\\)，量化**闭合误差 vs. 摄动强度**。  
- **多目标**：放宽脉冲上界或加入低推力段（伪谱/分段恒推），观察 **Pareto 前沿拓展**与 **HV** 变化。  
- **OD 可辨识度**：加/不加机动、加/不加 SRP 的测量仿真，对比 **CRB/2σ** 体积。

---

## 8. 工程化建议
- 将 **multiple shooting** 扩展为**全状态校正 + 分段 STM 串接**，并做**不变子空间一致性**检验（\\(\|\sin\angle(\mathcal{E}^{s,u}_{\rm num},\mathcal{E}^{s,u}_{\rm ref})\|\\) 曲线）。  
- 用稀疏线性代数实现 **Hermite–Simpson NLP** 的 KKT 牛顿步，绘制**KKT 残差**与**谱半径**对比。  
- 将鲁棒性目标从终端偏差扩展为 **灵敏度范数**（如 \\(\|\Phi(T)\|\_2\\) 或某方向投影），并在 **NSGA‑II** 与 **梯度法** 间对比收敛与稳定性。  
- OD 中引入 **UKF/批处理白化** 与 **光谱残差分析**，展示系统误差（未建模 SRP/J2/机动）对可观测性的影响。

---

## 9. 生成代码与结构小结
- **STM 真导数**：在 `integrators/propagate_with_stm.py` 中共同传播 \\(\mathbf{x},\Phi\\)；\\(\dot\Phi=A(\mathbf{x})\Phi\\)。  
- **单子矩阵**：`monodromy(x0,T,...)` 直接返回 \\(\Phi(T)\\)。  
- **修正方法**：`halo/diff_correction.py` 与 `solvers/*`；**多段**与**配点**可并列对比。  
- **优化**：`mission/transfer_opt.py` 实现轻量 **NSGA‑II**；前沿与超体积在 `plots/plotting.py` 可视化。  
- **定轨**：`od/estimation.py` 的 Batch LS/EKF 与 CRB。
<img width="960" height="640" alt="fig_02_streamplot" src="https://github.com/user-attachments/assets/fe547426-f891-4a7d-bca9-69604744a0a7" />
<img width="960" height="768" alt="fig_03_halo3d" src="https://github.com/user-attachments/assets/141c5a38-3d4d-4fd0-b6a8-60a1d940a257" />
<img wi<img width="880" height="832" alt="fig_10_polar_heat" src="https://github.com/user-attachments/assets/d08f58dc-5c39-41c0-9c22-8ea9cec15cf9" />
dth="960" height="640" alt="fig_06_drift_heatmap" src="https://github.com/user-attachments/assets/531bb041-9b49-46de-9ef4-6bb7e90b9853" />
<img w<img width="800" height="800" alt="fig_15_eigs_complex" src="https://github.com/user-attachments/assets/42f88c88-2110-4627-bbe0-022d41223117" />
idth="960" height="736" alt="fig_11_hv_surface" src="https://github.com/user-attachments/assets/c9ceabc9-53c7-4347-9242-3d808971e17c" />

---

### 引用与致谢
本项目为教学/研究演示实现，公式与思路取材于经典深空力学、变分法、数值优化与估计理论的通用框架；工程化与更高保真度实现（如非圆三体、真实 SRP 几何、严格 J2 在旋转系的推导、稀疏 KKT 求解等）可在此基础上继续拓展。
