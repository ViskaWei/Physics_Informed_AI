# BOSZ50k Z1 数据生成任务汇总

## 任务概述

- **任务类型**: PFS 光谱模拟数据生成
- **模型**: BOSZ 恒星大气模型 (R=50,000)
- **仪器**: PFS Medium Resolution (MR) arm
- **开始时间**: 2024-12-23 21:13
- **更新时间**: 2024-12-26

---

## 数据路径

### 源数据
```
模型网格: ${PFSSPEC_DATA}/models/stellar/grid/bosz/bosz_50000
```

### 输出数据
```
主目录: /datascope/subaru/user/swei20/data/bosz50000/z1/mag205_225_lowT_1M/

训练数据:
  - train_200k_0/dataset.h5  (6.8G, 200k 样本)
  - train_200k_1/dataset.h5  (6.8G, 200k 样本)
  - train_200k_2/dataset.h5  (6.8G, 200k 样本)
  - train_200k_3/dataset.h5  (6.8G, 200k 样本)
  - train_200k_4/dataset.h5  (6.8G, 200k 样本)

测试数据:
  - test_1k_0/
  - test_1k_1/

日志:
  - logs/shard_0.log ~ shard_4.log
```

---

## 当前进度 (2024-12-26 22:15 更新)

| Shard | 进度 | 已完成 | 总数 | 状态 |
|-------|------|--------|------|------|
| 0 | 35% | 70,012 | 200,000 | 🔄 运行中 |
| 1 | 36% | 71,013 | 200,000 | 🔄 运行中 |
| 2 | 35% | 70,011 | 200,000 | 🔄 运行中 |
| 3 | 35% | 70,012 | 200,000 | 🔄 运行中 |
| 4 | 36% | 71,012 | 200,000 | 🔄 运行中 |

**总进度**: ~352,060 / 1,000,000 样本 (**~35%**)

**预计完成时间**: 约 5-7 天后 (130-170 小时)

---

## 参数配置

### 恒星参数范围
| 参数 | 范围 | 分布 |
|------|------|------|
| T_eff (有效温度) | 3750 - 6000 K | beta |
| log_g (表面重力) | 1.0 - 5.0 | beta |
| M_H (金属丰度) | -2.5 - 0.75 | beta |

### 观测参数
| 参数 | 范围/值 | 说明 |
|------|---------|------|
| mag (星等) | 20.5 - 22.5 | HSC i-band, uniform 分布 |
| z (红移) | -0.001 - 0.001 | uniform 分布 |
| seeing | 0.5 - 1.5 arcsec | |
| exp_count | 12 | 曝光次数 |
| exp_time | 900s | 单次曝光时间 |
| target_zenith_angle | 0 - 45° | |
| target_field_angle | 0 - 0.65° | |
| moon_phase | 0 | 新月 |

### 仪器配置
```
detector: ${PFSSPEC_DATA}/subaru/pfs/arms/mr.json
detector_psf_pca: ${PFSSPEC_DATA}/subaru/pfs/psf/import/mr.2/pca.h5
sky: ${PFSSPEC_DATA}/subaru/pfs/noise/import/sky/mr/sky.h5
moon: ${PFSSPEC_DATA}/subaru/pfs/noise/import/moon/mr/moon.h5
model_res: 50000
```

### 数据处理
| 参数 | 值 |
|------|-----|
| norm | median |
| norm_wave | 6500 - 9500 Å |
| wave_resampler | rebin |
| interp_mode | spline |
| sample_mode | random |
| redden | true |

---

## 运行命令

```bash
python -m pfs.ga.pfsspec.sim.scripts.sim model bosz pfs \
    --threads 12 \
    --config /datascope/subaru/user/swei20/data/bosz50000/z1/train.json \
             /datascope/subaru/user/swei20/data/bosz50000/z1/inst_pfs_mr.json \
    --out /datascope/subaru/user/swei20/data/bosz50000/z1/mag205_225_lowT_1M/train_200k_X \
    --sample-count 200000 \
    --seeing 0.5 1.5
```

---

## 配置文件

### train.json
```
/datascope/subaru/user/swei20/data/bosz50000/z1/train.json
```

### inst_pfs_mr.json
```
/datascope/subaru/user/swei20/data/bosz50000/z1/inst_pfs_mr.json
```

---

## 监控命令

```bash
# 查看进度
for i in 0 1 2 3 4; do
    echo "=== Shard $i ===" 
    tail -1 /datascope/subaru/user/swei20/data/bosz50000/z1/mag205_225_lowT_1M/logs/shard_$i.log | grep -oP '\d+%|\d+/\d+'
done

# 查看进程
ps aux | grep "pfs.ga.pfsspec.sim" | grep -v grep
```

---

## 备注

- 数据集描述: **mag205_225_lowT_1M** - 暗星等 (20.5-22.5), 低温恒星 (3750-6000K), 共 1M 样本
- 5 个并行 shard, 每个生成 200k 样本
- 使用 12 线程并行处理
- 数据用于训练 Physics-Informed AI 模型

