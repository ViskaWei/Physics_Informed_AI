# 🤖 实验 Coding Prompt: ViT Scaling Curve

> **日期:** 2025-12-27 | **来源:** `logg/vit/exp_vit_scaling_curve_20251227.md`  
> **Experiment ID:** `VIT-20251227-vit-scaling-curve-01`  
> **MVP:** MVP-3.0 | **Project:** VIT

---

## ⚠️ 核心规则

| 规则 | 说明 |
|------|------|
| **nohup 后台运行** | 所有训练必须 `nohup ... &`，>20分钟不持续追踪 |
| **跨仓库用终端** | 写入 Physics_Informed_AI 用 `cat/echo/cp`，禁止 IDE 工具 |
| **图片必须入报告** | 所有图表必须在报告 §3 中引用，路径 `logg/vit/exp/img/` |
| **语言** | Header 英文 \| 正文中文 \| 图表文字英文 |

---

## 🚀 仓库路由

| Topic | 仓库 | 前缀 |
|-------|------|------|
| vit | `~/VIT` | VIT- |

---

## 📋 执行流程

### Step 1: 准备数据子集

**数据路径**: `/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/`

**需要生成的数据子集**:
- 50k: 从 `train_200k_0` 前 50k 样本
- 100k: 从 `train_200k_0` 前 100k 样本
- 200k: 使用 `train_200k_0` 完整 shard（已有）
- 500k: 使用 `train_200k_0` + `train_200k_1` + `train_200k_2` 前 100k

**参考脚本**: 查看 `~/VIT/scripts/` 中是否有数据子集生成脚本，或参考 `exp_vit_1m_scaling` 的数据加载方式

### Step 2: 创建配置文件

**需要 4 个配置文件**（每个数据规模一个）:

```yaml
# configs/exp/vit_scaling_50k.yaml
experiment_id: "VIT-20251227-vit-scaling-curve-01-50k"
data:
  source: "BOSZ"
  path: "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/"
  train_shards: ["train_200k_0"]  # 前 50k
  train_size: 50000
  val_size: 1000
  test_size: 10000
  noise_level: 1.0
  noise_type: "heteroscedastic"
  apply_noise_to: "train"  # on-the-fly for train, fixed for val/test

model:
  type: "ViT"
  image_size: 4096
  patch_size: 16
  hidden_size: 256
  num_hidden_layers: 6
  num_attention_heads: 8
  intermediate_size: 1024
  proj_fn: "C1D"  # Conv1D tokenization
  position_encoding: "learned"
  head_dim: 32
  dropout: 0.1

training:
  loss: "MSE"
  label_norm: "standard"  # z-score normalization
  optimizer: "AdamW"
  lr: 0.0003
  weight_decay: 0.0001
  lr_scheduler: "cosine"
  eta_min: 1e-5
  epochs: 200
  batch_size: 256
  gradient_clip: 0.5
  precision: "16-mixed"
  seed: 42
  early_stopping:
    monitor: "val_r2"
    mode: "max"
    patience: 20
```

**同样创建**: `vit_scaling_100k.yaml`, `vit_scaling_200k.yaml`, `vit_scaling_500k.yaml`（只改 `train_size` 和 `train_shards`）

### Step 3: 启动训练（4 个实验）

**使用训练驱动器**（推荐）:

```bash
cd ~/VIT && source init.sh

# 50k 实验
python /home/swei20/Physics_Informed_AI/_backend/scripts/training/driver.py \
    --config configs/exp/vit_scaling_50k.yaml \
    --exp-id VIT-20251227-vit-scaling-curve-01-50k \
    --work-dir ~/VIT \
    --health-time 600

# 100k 实验
python /home/swei20/Physics_Informed_AI/_backend/scripts/training/driver.py \
    --config configs/exp/vit_scaling_100k.yaml \
    --exp-id VIT-20251227-vit-scaling-curve-01-100k \
    --work-dir ~/VIT \
    --health-time 600

# 200k 实验
python /home/swei20/Physics_Informed_AI/_backend/scripts/training/driver.py \
    --config configs/exp/vit_scaling_200k.yaml \
    --exp-id VIT-20251227-vit-scaling-curve-01-200k \
    --work-dir ~/VIT \
    --health-time 600

# 500k 实验
python /home/swei20/Physics_Informed_AI/_backend/scripts/training/driver.py \
    --config configs/exp/vit_scaling_500k.yaml \
    --exp-id VIT-20251227-vit-scaling-curve-01-500k \
    --work-dir ~/VIT \
    --health-time 600
```

**或使用 nohup 后台运行**（如果驱动器不可用）:

```bash
cd ~/VIT && source init.sh

# 50k 实验
nohup python scripts/train_vit.py \
    --config configs/exp/vit_scaling_50k.yaml \
    > logs/VIT-20251227-vit-scaling-curve-01-50k.log 2>&1 &
echo $! > logs/VIT-20251227-vit-scaling-curve-01-50k.pid

# 100k 实验
nohup python scripts/train_vit.py \
    --config configs/exp/vit_scaling_100k.yaml \
    > logs/VIT-20251227-vit-scaling-curve-01-100k.log 2>&1 &
echo $! > logs/VIT-20251227-vit-scaling-curve-01-100k.pid

# 200k 实验
nohup python scripts/train_vit.py \
    --config configs/exp/vit_scaling_200k.yaml \
    > logs/VIT-20251227-vit-scaling-curve-01-200k.log 2>&1 &
echo $! > logs/VIT-20251227-vit-scaling-curve-01-200k.pid

# 500k 实验
nohup python scripts/train_vit.py \
    --config configs/exp/vit_scaling_500k.yaml \
    > logs/VIT-20251227-vit-scaling-curve-01-500k.log 2>&1 &
echo $! > logs/VIT-20251227-vit-scaling-curve-01-500k.pid
```

**确认正常后输出**:
```
✅ 任务已启动 (PID: xxx)
📋 tail -f ~/VIT/logs/VIT-20251227-vit-scaling-curve-01-50k.log
⏱️ 预计每个实验 ~2-4 小时，完成后告诉我继续
```

### Step 4: 收集结果

**等待所有 4 个实验完成后，收集结果**:

```python
# scripts/collect_vit_scaling_results.py
import json
from pathlib import Path

results = {}
for size in [50, 100, 200, 500]:
    exp_id = f"VIT-20251227-vit-scaling-curve-01-{size}k"
    result_path = Path(f"~/VIT/results/{exp_id}/summary.json")
    
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
            results[f"{size}k"] = {
                "test_r2": data.get("test_r2"),
                "test_mae": data.get("test_mae"),
                "best_epoch": data.get("best_epoch"),
            }
    
# 保存汇总结果
with open("~/VIT/results/vit_scaling_summary.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Step 5: 生成图表

**主图：ViT vs Traditional ML Scaling Curve**

```python
# scripts/plot_vit_scaling_curve.py
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 读取 ViT 结果
with open("~/VIT/results/vit_scaling_summary.json") as f:
    vit_results = json.load(f)

# 传统 ML 数据（已有）
ml_data = {
    "50k": {"Ridge": 0.4419, "LightGBM": 0.4879},
    "100k": {"Ridge": 0.4753, "LightGBM": 0.5533},
    "200k": {"Ridge": 0.4738, "LightGBM": 0.5466},
    "500k": {"Ridge": 0.4898, "LightGBM": 0.5743},
}

# 数据规模（log scale）
sizes = [50, 100, 200, 500]
sizes_log = np.log10(sizes)

# 提取 R² 值
vit_r2 = [vit_results[f"{s}k"]["test_r2"] for s in sizes]
ridge_r2 = [ml_data[f"{s}k"]["Ridge"] for s in sizes]
lgbm_r2 = [ml_data[f"{s}k"]["LightGBM"] for s in sizes]

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sizes_log, vit_r2, 'o-', color='#FFA500', label='ViT (p16_h256_L6)', linewidth=2, markersize=8)
ax.plot(sizes_log, lgbm_r2, 's-', color='#2E8B57', label='LightGBM', linewidth=2, markersize=8)
ax.plot(sizes_log, ridge_r2, '^-', color='#4169E1', label='Ridge', linewidth=2, markersize=8)

ax.set_xlabel('Dataset Size (log scale)', fontsize=12)
ax.set_ylabel('Test R²', fontsize=12)
ax.set_title('ViT vs Traditional ML Scaling Curve', fontsize=14, fontweight='bold')
ax.set_xticks(sizes_log)
ax.set_xticklabels([f'{s}k' for s in sizes])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.3, 0.75])

# 标注关键点
for i, (s, vit, lgbm) in enumerate(zip(sizes, vit_r2, lgbm_r2)):
    if vit > lgbm:
        ax.annotate(f'ViT > LGBM\n@{s}k', 
                   xy=(sizes_log[i], vit), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('/home/swei20/Physics_Informed_AI/logg/vit/exp/img/vit_scaling_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/swei20/Physics_Informed_AI/logg/vit/exp/img/vit_scaling_curve.pdf', bbox_inches='tight')
print("✅ 图表已保存: logg/vit/exp/img/vit_scaling_curve.png")
```

### Step 6: 更新实验报告

**使用终端命令写入报告**（跨仓库规则）:

```bash
# 读取结果并更新报告
cat << 'EOF' > /home/swei20/Physics_Informed_AI/logg/vit/exp_vit_scaling_curve_20251227.md
[根据 exp.md 模板填写，包含：
- §1 目标
- §2 实验设计
- §3 实验图表（引用 vit_scaling_curve.png）
- §4 关键洞见
- §5 结论
- §6 附录（数值结果表）]
EOF
```

---

## 🗂️ 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `~/VIT/scripts/train_vit.py` | 训练主脚本 | 修改数据加载部分（支持子集） |
| `~/VIT/src/base/vit.py` | ViT 模型定义 | 无需修改（使用 p16_h256_L6_a8） |
| `~/VIT/src/data/dataset.py` | 数据加载器 | 可能需要修改以支持数据子集 |
| `logg/vit/exp_vit_1m_scaling_20251226.md` | 训练配置参考 | 参考训练参数设置 |

---

## 🎯 实验规格

```yaml
experiment_id: "VIT-20251227-vit-scaling-curve-01"
repo_path: "~/VIT"
data: 
  source: "BOSZ 50000, mag205_225_lowT_1M"
  path: "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M/"
  sizes: [50k, 100k, 200k, 500k]
  val_size: 1000
  test_size: 10000
noise: 
  sigma: 1.0
  type: "heteroscedastic"
  apply_to: "train"  # on-the-fly for train, fixed for val/test
model: 
  type: "ViT"
  architecture: "p16_h256_L6_a8"  # 固定架构
  patch_size: 16
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  proj_fn: "C1D"
training: 
  epochs: 200
  batch_size: 256
  lr: 3e-4
  optimizer: "AdamW"
  lr_scheduler: "cosine"
  loss: "MSE"
  label_norm: "standard"
  seed: 42
plots: 
  - type: scaling_curve
    save: "logg/vit/exp/img/vit_scaling_curve.png"
    compare_with: ["Ridge", "LightGBM"]
```

---

## ✅ 检查清单

- [ ] 数据子集已生成（50k, 100k, 200k, 500k）
- [ ] 4 个配置文件已创建
- [ ] 4 个训练任务已启动（nohup 后台运行）
- [ ] 所有训练已完成（检查 summary.json）
- [ ] 结果已收集（vit_scaling_summary.json）
- [ ] 图表已生成（vit_scaling_curve.png，英文标签）
- [ ] 图表已在报告 §3 中引用
- [ ] 报告已更新（包含数值结果表）
- [ ] 报告已同步到 roadmap.md（如有重要发现）

---

## 🔧 故障排除

| 问题 | 修复 |
|------|------|
| NaN loss | 降 lr / grad_clip / 检查数据 |
| OOM | 减 batch_size / 使用 gradient checkpointing |
| Loss 爆炸 | 降 lr / warmup / 检查 label norm |
| 数据加载错误 | 检查数据子集路径和索引 |
| 训练不收敛 | 检查 epochs 是否足够 / early stop 设置 |

---

## 📊 预期结果

**关键指标**:
- ViT @ 50k: R² ≈ 0.45-0.50（可能低于 LightGBM）
- ViT @ 100k: R² ≈ 0.55-0.60（接近 LightGBM）
- ViT @ 200k: R² ≈ 0.60-0.65（可能超越 LightGBM）
- ViT @ 500k: R² ≈ 0.65-0.70（显著超越 LightGBM）

**关键观察点**:
- ViT 何时超越 LightGBM？（预期在 200k-500k 之间）
- ViT scaling 斜率 vs 传统 ML（预期 ViT 斜率更大）

---

## 🔗 相关文件

| 类型 | 路径 | 说明 |
|------|------|------|
| 📗 实验报告 | `logg/vit/exp_vit_scaling_curve_20251227.md` | 主报告 |
| 🗺️ Roadmap | `logg/vit/vit_roadmap_20251227.md` | MVP-3.0 规格 |
| 🧠 Hub | `logg/vit/vit_hub_20251227.md` | 战略导航 |
| 📗 传统 ML Scaling | `logg/scaling/exp/exp_scaling_ml_ceiling_20251222.md` | 基线对比数据 |
| 📊 图表输出 | `logg/vit/exp/img/vit_scaling_curve.png` | 主图保存位置 |

---

*Generated: 2025-12-27 | Status: 🔄 待执行*
