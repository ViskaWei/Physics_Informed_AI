# 📋 归档队列 (Archive Queue)

> **使用方法**: 在 Cursor 中输入 `a` 开始归档

---

## ⏳ 待归档 (Pending)

| 源文件 | 目标目录 | 优先级 |
|--------|----------|--------|
| - | - | - |

> 💡 `raw/` 目录当前为空，新实验结果放入后会自动出现在此

---

## ✅ 已归档 (Archived)

| 源文件 | 归档报告 | 归档日期 |
|--------|----------|----------|
| `raw_vit/CNN_DILATED_EXPERIMENTS_FULL_REPORT.md` | `logg/cnn/exp_cnn_dilated_kernel_sweep_20251201.md` | 2025-12-01 |
| `raw_blindspot/layer_pooling_experiment_report.md` | `logg/distill/exp_error_info_decomposition_20251201.md` | 2025-12-01 |
| `raw_blindspot/linear_probe_report.md` | `logg/distill/exp_linear_probe_latent_20251130.md` | 2025-11-30 |

---

## 📊 统计

- **待归档**: 0
- **已归档**: 3+
- **最后更新**: 2025-12-01

---

## 📁 目录结构

```
raw/                    ← 新实验结果放这里
    ↓ 归档
logg/[topic]/exp_*.md   ← 结构化报告
    ↓ 原文件移动
processed/raw/          ← 已处理的原始文件
```

> **快捷命令**: `a` 归档 | `s` 状态
