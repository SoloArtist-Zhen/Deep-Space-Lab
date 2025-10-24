
# Ultra-Fast Deep-Space Attitude Control Lab — 详细说明（含 $\LaTeX$ 数学公式）

> 面向“深空探测器姿态控制”的**极速可复现**研究底座。聚焦 **MPC + 自适应** 与 **LQR / H∞-like** 的**算法对比**，在**复杂空间环境**（热致惯量变化、SRP 扭矩代理、辐射噪声）与**故障模式**（传感器偏置跳变、执行器饱和/轴卡滞）下进行快速仿真，**一次运行生成 15 张高级图**。  

运行：
```bash
python -u run_attitude_fast15.py
```
图片输出目录：`attitude_ultrafast/outputs/`。

---

## 1. 动力学模型（Quaternion + 刚体转动）

**四元数运动学**
\[
\dot{\mathbf{q}} \;=\; \tfrac{1}{2}\,\mathbf{q}\otimes \begin{bmatrix}0\\ \boldsymbol\omega\end{bmatrix},
\qquad
\mathbf{q}=\begin{bmatrix}q_w \\ q_x \\ q_y \\ q_z\end{bmatrix},\ \ \|\mathbf{q}\|=1.
\]

**刚体动力学（惯量 $J$ 可能随温度缓慢变化）**
\[
J(t)\,\dot{\boldsymbol\omega} \;+\; \boldsymbol\omega \times \bigl(J(t)\,\boldsymbol\omega\bigr)
\;=\; \boldsymbol\tau_c \;+\; \boldsymbol\tau_d,
\]
其中 $\boldsymbol\tau_c$ 为控制力矩，$\boldsymbol\tau_d$ 为外部摄动力矩（如 SRP、热弹性等）。

**热致惯量变化（对角近似）**
\[
J(t) \;=\; \mathrm{diag}\!\Big(J_{x0}\,[1+\alpha_x\sin(\nu t+\phi_x)],\ 
J_{y0}\,[1+\alpha_y\sin(\nu t+\phi_y)],\ 
J_{z0}\,[1+\alpha_z\sin(\nu t+\phi_z)]\Big).
\]

**SRP 扭矩代理**（以太阳方向 $\hat{\mathbf{s}}$ 与偏心力臂 $\mathbf{r}_c$ 生成）
\[
\boldsymbol\tau_{\mathrm{SRP}} \;=\; c_{\mathrm{srp}}\,A\,\bigl(\mathbf{r}_c \times \hat{\mathbf{s}}_B\bigr),
\quad \hat{\mathbf{s}}_B = R_{IB}(\mathbf{q})^\top \hat{\mathbf{s}}_I.
\]

> 实现上，积分器采用**半隐式欧拉**：先更新角速度再指数映射更新四元数，保证数值稳定与 $\|\mathbf{q}\|=1$ 归一。代码见 `core/dynamics.py::step_attitude`。

---

## 2. 误差表示与离散化

**姿态误差四元数**
\[
\mathbf{q}_e \;=\; \mathbf{q}_d^{-1}\otimes \mathbf{q}, \qquad
\text{小角度误差向量}\ \ \mathbf{e}\ \approx\ 2\,\mathrm{vec}(\mathbf{q}_e)\cdot \mathrm{sgn}(q_{e,w}).
\]

小角度线性化得到**误差状态** $\mathbf{x}=[\mathbf{e}^\top,\ \boldsymbol\omega^\top]^\top$ 的连续近似：
\[
\dot{\mathbf{e}} \approx \boldsymbol\omega,\qquad
J\,\dot{\boldsymbol\omega} \approx \boldsymbol\tau_c + \boldsymbol\tau_d.
\]

**一阶保持离散化**（采样时间 $T_s$）：
\[
\mathbf{x}_{k+1} \;=\; A_d\,\mathbf{x}_k + B_d\,\mathbf{u}_k,\quad
A_d=\begin{bmatrix}I & T_s I\\ 0 & I\end{bmatrix},\ 
B_d=\begin{bmatrix}0\\ T_s J^{-1}\end{bmatrix}.
\]

---

## 3. 传感器/执行器故障模型

- **陀螺偏置跳变**：
  \[ \tilde{\boldsymbol\omega}(t)=\boldsymbol\omega(t)+\mathbf{b}(t)+\boldsymbol\eta,\quad
  \mathbf{b}(t)=\sum_i \mathbf{b}_i\,\mathbf{1}_{t\ge t_i}. \]
- **执行器饱和**：
  \[ \mathbf{u}=\mathrm{sat}(\mathbf{u}_{\mathrm{cmd}},\ \mathbf{u}_{\max}). \]
- **轴卡滞**：某一轴 $u_j\equiv 0$。

这些故障/约束在 `run_attitude_fast15.py::run_episode` 中统一注入。

---

## 4. 控制策略

### 4.1 LQR（离散）
目标
\[
\min_{\{\mathbf{u}_k\}}\ \sum_{k=0}^{N-1}\bigl(\mathbf{x}_k^\top Q\,\mathbf{x}_k
+ \mathbf{u}_k^\top R\,\mathbf{u}_k\bigr)\ +\ \mathbf{x}_N^\top P\,\mathbf{x}_N
\quad\text{s.t.}\quad \mathbf{x}_{k+1}=A_d\mathbf{x}_k+B_d\mathbf{u}_k.
\]
解由 Riccati 方程给出并形成**恒定增益** $\mathbf{u}=-K\mathbf{x}$。

### 4.2 H∞-like 稳健调权（轻量代理）
为快速对比稳健性，使用放大 $Q$、收缩 $R$ 的“**H∞-like**”代理（非严格 H∞），提升状态抑制与阻尼：
\[
Q'=\alpha Q,\ R'=\beta R,\ \alpha>1,\ 0<\beta\le 1.
\]

### 4.3 MPC（有限时域预览 + 投影）
采用**无约束 LQR 预览**（时变增益序列 $K_k$）给出 $u_0$，再对 $\mathrm{sat}(\cdot)$ 投影以满足饱和：
\[
J = \sum_{k=0}^{N_p-1}\!\bigl(\mathbf{x}_k^\top Q\mathbf{x}_k + \mathbf{u}_k^\top R\mathbf{u}_k\bigr)
+ \mathbf{x}_{N_p}^\top P\mathbf{x}_{N_p},\qquad \mathbf{u}_0 = -K_0\,\mathbf{x}_0,\ 
\mathbf{u}=\mathrm{sat}(\mathbf{u}_0).
\]

> 端管（terminal tube）用**椭圆集**近似：
\[
\mathcal{E}=\bigl\{\mathbf{e}\in\mathbb{R}^3\ \big|\ \mathbf{e}^\top P_{\mathrm{tube}}\mathbf{e}\le 1\bigr\},
\]
并以图形方式呈现其约束形状（图 6）。

### 4.4 自适应增益（对角参数 $\boldsymbol\theta$）
对 MPC 力矩加入**自适应项**：
\[
\mathbf{u} \;=\; \mathbf{u}_{\mathrm{MPC}} \;+\; \Theta\,\boldsymbol\omega,\qquad
\Theta=\mathrm{diag}(\boldsymbol\theta).
\]
参数更新（离散 $\sigma$-修正）：
\[
\boldsymbol\theta_{k+1}\;=\;\boldsymbol\theta_k\;+\;\gamma\,\Phi(\mathbf{x}_k)\,\mathbf{e}_k\;-\;\sigma\,\boldsymbol\theta_k,
\]
其中 $\Phi$ 为回归矩阵（本实现取 $\Phi=\mathrm{diag}(|\boldsymbol\omega|+\varepsilon)$），$\gamma>0$ 为学习率，$\sigma>0$ 为抑制漂移的 $\sigma$-修正。

> **稳定性直觉**：选取复合 Lyapunov 候选
\[
V = \mathbf{x}^\top P \mathbf{x} + \tfrac{1}{\gamma}\,(\boldsymbol\theta-\boldsymbol\theta^\star)^\top(\boldsymbol\theta-\boldsymbol\theta^\star),
\]
在 $\sigma$-修正下保证参数有界，闭环误差能量下降（详见 MRAC/L1 相关理论；此处给出轻量近似实现以维持**极速可跑**）。

---

## 5. 评测指标与“顶会级”图形

**误差峰值**（越小越好）：
\[
E_{\infty} \;=\; \mathrm{quantile}_{0.95}\bigl(\|\mathbf{e}(t)\|\bigr).
\]

**控制能量**（离散求和的二范数）：
\[
E_u \;=\; \sum_k \|\mathbf{u}_k\|_2^2.
\]

**故障鲁棒性**（故障场景误差的高分位）：
\[
R_{\mathrm{fault}} \;=\; \mathrm{quantile}_{0.95}\bigl(\|\mathbf{e}(t)\|\ \text{under faults}\bigr).
\]

这三者组成**三目标**“峰值误差/能量/鲁棒性”的**Pareto 前沿**（图 5）。

---

## 6. 复现实验清单（15 张高级图）

> 对应 `run_attitude_fast15.py` 的输出文件，均在 `attitude_ultrafast/outputs/`。

1. **fig01\_sphere\_pointing.png**：单位球上指向轨迹（四算法叠加，颜色区分）。  
2. **fig02\_triad\_frames.png**：初/中/末三时刻的机体坐标轴三元组（3D 箭簇）。  
3. **fig03\_sphere\_heat.png**：Lambert 球面热力图（随时间权重衰减）。  
4. **fig04\_polar\_torque\_LQR.png / \_MPC\_ADAPT.png**：控制力矩极坐标密度（方向–幅值分布核密度）。  
5. **fig05\_pareto\_algos.png**：三目标 Pareto 3D（四算法各一点评估）。  
6. **fig06\_mpc\_tube.png**：MPC 端管（椭圆等高线示意）。  
7. **fig07\_eig\_locus.png**：闭环特征值复平面散点（惯量缩放族）。  
8. **fig08\_spectrogram.png**：误差 STFT “谱图”（MPC+自适应）。  
9. **fig09\_mc\_density.png**：Monte Carlo 终态指向球面密度（故障+扰动）。  
10. **fig10\_stream\_fields.png**：线性化误差场流线（LQR vs MPC）。  
11. **fig11\_fault\_mosaic.png**：故障时间轴马赛克热图（卡滞/偏置跳变）。  
12. **fig12\_theta\_traj.png**：自适应参数演化 3D 轨迹。  
13. **fig13\_saturation.png**：反作用轮饱和热图（轴×时间）。  
14. **fig14\_lyap.png**：Lyapunov 函数等高线（$(e_y,\omega_y)$ 平面）。  
15. **fig15\_margin\_polar.png**：稳定裕度极坐标密度（谱半径 proxy）。

---

## 7. 复现实验参数（默认）

- 采样时间：$T_s=0.05\ \mathrm{s}$，仿真时长 $T=15\ \mathrm{s}$。  
- 惯量标称：$J_0=\mathrm{diag}(0.12,\ 0.09,\ 0.08)\ \mathrm{kg\,m^2}$；热致幅度 $\alpha\in[0.1,0.3]$。  
- 饱和上限：$u_{\max}=[0.08,\ 0.06,\ 0.05]\ \mathrm{N\,m}$。  
- LQR：$Q=\mathrm{diag}(50,50,50,\,2,2,2)$，$R=\mathrm{diag}(0.6,0.6,0.6)$。  
- H∞-like：$\alpha=3.0,\ \beta=0.8$（对 $Q,R$ 的缩放）。  
- MPC：预测域 $N_p=10$，预览 LQR 时变增益，后投影至饱和。  
- 自适应：$\gamma=0.15,\ \sigma=0.05$，$\Phi=\mathrm{diag}(|\boldsymbol\omega|+\varepsilon)$。  
- 故障：$t=5\ \mathrm{s}$ 时陀螺偏置跳变，$y$ 轴执行器卡滞示例。

---

## 8. 与**顶会/顶刊**接轨的升级建议

1. **真约束 MPC（QP）**：用 PGD/ADMM 或 OSQP 将饱和/速率/端管约束显式化；对比“预览+投影”。  
2. **MRAC / L1 自适应**：加入投影算子与低通滤波，给出**严格稳定性证明**与域内鲁棒性界。  
3. **感知–控制闭环**：接入星敏/陀螺融合（EKF/UKF），报告**可观测性谱**、**CRB 下界**与闭环性能。  
4. **高保真摄动**：SRP 几何更真实，热–结构–光学耦合；加入**轮系/磁控**混合执行链。  
5. **故障诊断**：增设 FDI（残差生成/投票），并给出**误检/漏检 ROC 曲线**与**再配置控制**效果。  
6. **实验统计**：更大规模 Monte Carlo，报告 **Pareto 前沿**的**置信带/超体积**收敛。

---

## 9. 代码结构

```
attitude_ultrafast/
  core/dynamics.py          # 四元数/刚体/摄动与故障注入
  controllers/lqr_hinf.py   # LQR 与 H∞-like 轻量稳健调权
  controllers/mpc_adapt.py  # MPC 预览 + 饱和投影；自适应增益
  plots/advplots.py         # 球面/极坐标/流线/复平面/谱图等绘图工具
  run_attitude_fast15.py    # 一键生成 15 张高级图
  outputs/                  # 结果图片（运行后生成）
  README_detailed.md        # 本说明
```

---

## 10. 复现实用技巧

- 使用无交互后端：脚本中已 `matplotlib.use("Agg")`，确保**只出图不弹窗**。  
- 观察进度：`python -u` 以**取消缓冲**，实时打印。  
- 随机性：可固定 `seed` 重复结果（代码中示例 `seed=1/2/…`）。  
- 性能：本仓库定位“**秒级出图**”，如需更高精度，可逐步加长 $N_p$、减小 $T_s$、显式 QP。

---

## 11. 术语速查

- **端管（terminal tube）**：MPC 中用于保证末端/鲁棒性的终端可控集合，常以椭球/多面体近似。  
- **$\sigma$-修正**：自适应律中的泄露项，抑制参数漂移、提高有界性。  
- **CRB（Cramér–Rao Bound）**：估计方差下界，常用于观测系统可辨识度评估。

---

祝你开题顺利、图好看、结果硬！如需**论文模板版**（自动导出 PDF/EPS、按 A4/双栏列宽配置字体/线宽/标注），可在本仓库上继续扩展。
