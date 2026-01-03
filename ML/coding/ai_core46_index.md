# AI 核心 46 题（按日期与类别映射｜简化类别）

- 说明
  - 来源：`ML/AI_编程题_Python解答_核心46题.md`
  - 目的：为每题指定分类与 slug，统一落地到 `ML/coding/<tag>/...`
  - 简化类别（tag）：reg, cluster, tree, feature, conv, att, quant, prune, rnn, tokenizer, moe, dp, graph, window
  - 状态：未整合/已整合（链接到 main 文件）

## 📊 分类进度总览

| Tag | 题数 | 完成 | Main 文件 |
|-----|------|------|-----------|
| reg | 6 | 0 | [reg_main.md](./reg/reg_main.md) |
| cluster | 7 | 0 | [cluster_main.md](./cluster/cluster_main.md) |
| att | 6 | 0 | [att_main.md](./att/att_main.md) |
| feature | 6 | 0 | [feature_main.md](./feature/feature_main.md) |
| conv | 4 | 4 | [conv_main.md](./conv/conv_main.md) ✅ |
| tree | 4 | 0 | [tree_main.md](./tree/tree_main.md) |
| dp | 3 | 0 | [dp_main.md](./dp/dp_main.md) |
| quant | 2 | 0 | [quant_main.md](./quant/quant_main.md) |
| graph | 2 | 0 | [graph_main.md](./graph/graph_main.md) |
| rnn | 2 | 0 | [rnn_main.md](./rnn/rnn_main.md) |
| window | 2 | 0 | [window_main.md](./window/window_main.md) |
| prune | 1 | 0 | [prune_main.md](./prune/prune_main.md) |
| tokenizer | 1 | 0 | [tokenizer_main.md](./tokenizer/tokenizer_main.md) |
| moe | 1 | 0 | [moe_main.md](./moe/moe_main.md) |
| **总计** | **46** | **4** | - |

---

## 🎯 1/8 考试 Tag 优先级预测

> **分析日期**: 2026-01-02  
> **核心依据**: 「不连考」原则 + 「轮换周期」规律

### 历史出题时间线

| 日期 | Tag 1 | Tag 2 |
|------|-------|-------|
| 12-17 | reg | quant |
| 12-03 | prune | cluster |
| 11-20 | att | conv |
| 11-19 | cluster | graph |
| 11-12 | quant | tree |
| 11-06 | reg | conv |
| 11-05 | feature | dp |
| 10-29 | feature | reg |
| 10-23 | feature | conv |
| 10-22 | conv | att |
| 10-17 | cluster | rnn |
| 10-15 | att | cluster |
| 10-10 | cluster | rnn |
| 09-28 | cluster | att |
| 09-24 | cluster | tree |
| 09-18 | conv | dp |
| 09-17 | att | tokenizer |
| 09-12 | graph | att |
| 09-10 | window | window |
| 09-05 | tree | dp |
| 09-04 | feature | reg |
| 09-03 | moe | reg |
| 08-27 | feature | tree |

### 🔴 P0 - 必刷（3-4个月没考，极高概率）

| Tag | 上次出现 | 距今 | 题数 | 理由 |
|-----|---------|------|------|------|
| **window** | 09-10 | ~4个月 | 2题 | 最久没考，大概率轮到 |
| **tokenizer** | 09-17 | ~4个月 | 1题 | 久未出现，只有1题要掌握 |
| **moe** | 09-03 | ~4个月 | 1题 | 久未出现，路由问题必会 |

### 🟡 P1 - 重点复习（2个月没考，较高概率）

| Tag | 上次出现 | 距今 | 题数 | 理由 |
|-----|---------|------|------|------|
| **rnn** | 10-17 | ~2.5个月 | 2题 | LSTM/反向传播必考经典 |
| **graph** | 11-19 | ~1.5个月 | 2题 | 树/图题频率低但稳定出 |
| **dp** | 11-05 | ~2个月 | 3题 | 路径/决策类dp常考 |

### 🟢 P2 - 巩固（1个月左右，可能出）

| Tag | 上次出现 | 距今 | 题数 | 理由 |
|-----|---------|------|------|------|
| **tree** | 11-12 | ~2个月 | 4题 | 决策树系列稳定考 |
| **feature** | 11-05 | ~2个月 | 6题 | 题多，变形多，需熟练 |

### ⚪ P3 - 了解即可（12月刚考过，短期不太可能重复）

| Tag | 上次出现 | 理由 |
|-----|---------|------|
| att | 11-20 | 频率虽高但刚考过 |
| conv | 11-20 | 刚考过，但模板要熟 |
| cluster | 12-03 | 刚考过 |
| prune | 12-03 | 刚考过且只有1题 |
| reg | 12-17 | 最近刚考 |
| quant | 12-17 | 最近刚考 |

### 📊 规律总结

- 同一 tag 连续两场考的概率 **极低**
- 高频 tag（cluster/att/conv/reg/feature）约 **3-5周** 轮回
- 低频 tag（window/tokenizer/moe）约 **3-4个月** 出现一次
- **结论**: 1/8 大概率考 `window` / `tokenizer` / `moe` / `rnn` 中的 1-2 个 + 1个中频tag（tree/dp/graph/feature）

---

## 📝 1/8 复习计划（6天冲刺）

| 天数 | 日期 | 复习Tag | 题数 | 重点内容 |
|------|------|---------|------|----------|
| D-6 | 1/2 | **window** | 2 | 滑动窗口特征转换、历史窗口搜索 |
| D-5 | 1/3 | **tokenizer** + **moe** | 2 | 大模型分词、MOE路由优化 |
| D-4 | 1/4 | **rnn** | 2 | LSTM结构实现、反向传播 |
| D-3 | 1/5 | **graph** + **dp** | 5 | k祖先节点、最大值子树、路径DP |
| D-2 | 1/6 | **tree** + **feature** | 10 | 决策树剪枝、特征工程 |
| D-1 | 1/7 | 模板回顾 | - | conv/att 模板快速过一遍 |

---

## 📋 题目明细

| 日期 | P编号 | 题目 | 类别(tag) | slug | 状态 | 源 |
|------|------|------|------|------|------|------|
| 2025-12-17 | P4532 | 使用线性回归预测手机售价 | reg | linear_regression_phone_price | 已整合 | [reg_main](./reg/reg_main.md) |
| 2025-12-17 | P4533 | 模型量化最小误差 | quant | quantization_min_error | 已整合 | [quant_main](./quant/quant_main.md) |
| 2025-12-03 | P4518 | 基于剪枝的神经网络模型压缩 | prune | pruning_compression | 已整合 | [prune_main](./prune/prune_main.md) |
| 2025-12-03 | P4519 | 智能客户分群与新用户定位(KMeans均衡分区版) | cluster | kmeans_balanced_customer_segmentation | 已整合 | [cluster_main](./cluster/cluster_main.md) |
| 2025-11-20 | P4481 | ViT Patch Embedding层实现 | att | vit_patch_embedding | 已整合 | [att_main](./att/att_main.md) |
| 2025-11-20 | P4482 | 带Padding的卷积计算 | conv | conv_with_padding | 已整合 | [conv_main](./conv/conv_main.md) |
| 2025-11-19 | P4475 | 终端款型聚类识别 | cluster | terminal_model_clustering | 已整合 | [cluster_main](./cluster/cluster_main.md) |
| 2025-11-19 | P4476 | 最大值子树 | graph | max_value_subtree | 已整合 | [graph_main](./graph/graph_main.md) |
| 2025-11-12 | P4464 | 全连接层INT8非对称量化实现 | quant | fc_int8_asym | 已整合 | [quant_main](./quant/quant_main.md) |
| 2025-11-12 | P4465 | 决策树的QAM调制符合检测 | tree | qam_decision_tree_detection | 已整合 | [tree_main](./tree/tree_main.md) |
| 2025-11-06 | P4447 | 医疗诊断模型的训练与更新 | reg | medical_diagnosis_train_update | 已整合 | [reg_main](./reg/reg_main.md) |
| 2025-11-06 | P4448 | 卷积操作 | conv | conv_operation | 已整合 | [conv_main](./conv/conv_main.md) |
| 2025-11-05 | P4441 | 多目标推荐排序模型优化 | feature | multi_objective_ranking_optimization | 已整合 | [feature_main](./feature/feature_main.md) |
| 2025-11-05 | P4442 | 须从规矩出方圆 | dp | rule_to_round | 已整合 | [dp_main](./dp/dp_main.md) |
| 2025-10-29 | P4343 | 实体匹配结果合并问题 | feature | entity_matching_merge | 已整合 | [feature_main](./feature/feature_main.md) |
| 2025-10-29 | P4344 | 商品购买预测 | reg | purchase_prediction | 已整合 | [reg_main](./reg/reg_main.md) |
| 2025-10-23 | P4277 | 人脸关键点对齐 | feature | face_keypoint_alignment | 已整合 | [feature_main](./feature/feature_main.md) |
| 2025-10-23 | P4278 | 卷积结构实现 | conv | conv_structure_impl | 已整合 | [conv_main](./conv/conv_main.md) |
| 2025-10-22 | P4274 | 最大能量路径 | conv | max_energy_path | 已整合 | [conv_main](./conv/conv_main.md) |
| 2025-10-22 | P4275 | 基于空间连续块的稀疏注意力机制 | att | sparse_attention_block | 已整合 | [att_main](./att/att_main.md) |
| 2025-10-17 | P4238 | 预训练模型智能告警聚类与故障诊断 | cluster | pretrained_alarm_clustering | 已整合 | [cluster_main](./cluster/cluster_main.md) |
| 2025-10-17 | P4239 | 反向传播实现 | rnn | backprop_impl | 已整合 | [rnn_main](./rnn/rnn_main.md) |
| 2025-10-15 | P4227 | 动态注意力掩码调度问题 | att | dynamic_attention_mask_scheduling | 已整合 | [att_main](./att/att_main.md) |
| 2025-10-15 | P4228 | 基于二分KMeans的子网分割 | cluster | bisecting_kmeans_subnet_split | 已整合 | [cluster_main](./cluster/cluster_main.md) |
| 2025-10-10 | P3874 | 数据聚类及噪声点识别 | cluster | clustering_noise_detection | 已整合 | [cluster_main](./cluster/cluster_main.md) |
| 2025-10-10 | P3875 | 经典LSTM模型结构实现 | rnn | lstm_structure_impl | 已整合 | [rnn_main](./rnn/rnn_main.md) |
| 2025-10-10us | P3871 | 磁盘故障检测的特征工程 | feature | disk_failure_feature_engineering | 已整合 | [feature_main](./feature/feature_main.md) |
| 2025-10-10us | P3872 | 基于逻辑回归的意图分类器 | reg | logreg_intent_classifier | 已整合 | [reg_main](./reg/reg_main.md) |
| 2025-09-28 | P3842 | Yolo检测器中的anchor聚类 | cluster | anchor_kmeans | 已整合 | [cluster_main](./cluster/cluster_main.md) |
| 2025-09-28 | P3843 | Masked Multi-Head Self-Attention 实现 | att | masked_mhsa_impl | 已整合 | [att_main](./att/att_main.md) |
| 2025-09-24 | P3791 | 无线网络优化中的基站聚类分析 | cluster | base_station_clustering | 已整合 | [cluster_main](./cluster/cluster_main.md) |
| 2025-09-24 | P3792 | 基于决策树的无线状态预测 | tree | wireless_state_decision_tree | 已整合 | [tree_main](./tree/tree_main.md) |
| 2025-09-18 | P3718 | 最大能量路径 | conv | max_energy_path | 已整合 | [conv_main](./conv/conv_main.md) |
| 2025-09-18 | P3719 | 数据中心水温调节档位决策 | dp | water_temp_gear_decision | 已整合 | [dp_main](./dp/dp_main.md) |
| 2025-09-17 | P3712 | 大模型Attention模块开发 | att | llm_attention_module | 已整合 | [att_main](./att/att_main.md) |
| 2025-09-17 | P3713 | 大模型分词 | tokenizer | llm_tokenizer | 已整合 | [tokenizer_main](./tokenizer/tokenizer_main.md) |
| 2025-09-12 | P3657 | 二叉树中序遍历的第k个祖先节点 | graph | kth_ancestor | 已整合 | [graph_main](./graph/graph_main.md) |
| 2025-09-12 | P3658 | 支持LoRA的Attention实现 | att | lora_attention_impl | 已整合 | [att_main](./att/att_main.md) |
| 2025-09-10 | P3639 | 历史的窗口搜索 | window | history_window_search | 已整合 | [window_main](./window/window_main.md) |
| 2025-09-10 | P3640 | 多尺寸窗口滑动的特征转换 | window | multi_size_window_transform | 已整合 | [window_main](./window/window_main.md) |
| 2025-09-05 | P3528 | 阈值最优的决策树 | tree | decision_tree_threshold_f1_opt | 已整合 | [tree_main](./tree/tree_main.md) |
| 2025-09-05 | P3529 | 随机游走问题 | dp | random_walk_problem | 已整合 | [dp_main](./dp/dp_main.md) |
| 2025-09-04 | P3561 | 大模型训练数据均衡分配算法 | feature | balanced_data_allocation | 已整合 | [feature_main](./feature/feature_main.md) |
| 2025-09-04 | P3562 | 传感器数据分析 | reg | sensor_data_analysis | 已整合 | [reg_main](./reg/reg_main.md) |
| 2025-09-03 | P3553 | 大模型训练MOE场景路由优化算法 | moe | moe_routing_optimization | 已整合 | [moe_main](./moe/moe_main.md) |
| 2025-09-03 | P3552 | 云存储设备故障预测 | reg | cloud_storage_failure_prediction | 已整合 | [reg_main](./reg/reg_main.md) |
| 2025-08-27 | P3479 | 标签样本数量 | feature | label_sample_count | 已整合 | [feature_main](./feature/feature_main.md) |
| 2025-08-27 | P3480 | F1值最优的决策树剪枝 | tree | decision_tree_pruning_f1 | 已整合 | [tree_main](./tree/tree_main.md) |

---

# 模版

### I/O
```python
import sys
data = sys.stdin.read().strip().split()
it = iter(data)
K = int(next(it)); C = R = int(next(it));
Ker = [[ int(next(it)) for _ in range(K)] for _ in range(K)]
Img = [[ int(next(it)) for _ in range(C)] for _ in range(R)]
...
sys.stdout.write("\n".join(" ".join(map(str, row)) for row in E))
```

### Conv 
```python
    k2 = K // 2;
    E = [[0.0] * C for _ in range(R)]
    Img_pad = [[0] * (C + 2 * k2) for _ in range(R + 2 * k2)]
    for r in range(R): Img_pad[r+k2][k2:k2+C] = Img[r][:] # r+k2 别忘
    for r in range(R):
        for c in range(C):
            summ = 0
            for kr in range(K):
                for kc in range(K):
                    summ += Img_pad[r+kr][c+kc] * Ker[kr][kc]
            E[r][c] = summ
```
