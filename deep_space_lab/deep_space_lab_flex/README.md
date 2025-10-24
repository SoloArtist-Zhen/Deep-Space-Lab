
# Deep-Space Lab (FLEX Edition)
三档速度：`--ultra`（最快）、`--mode fast`（默认）、`--mode precise`（更精）。仅依赖 numpy、matplotlib。

## 用法
```bash
# 全流程（最快）
python run_all.py --ultra

# 只跑某一部分（比如 Halo），最快
python run_all.py halo --ultra

# 快速但更稳
python run_all.py --mode fast

# 精度版（更慢但更细）
python run_all.py --mode precise
```

输出在：`deep_space_lab_flex/outputs/`。
