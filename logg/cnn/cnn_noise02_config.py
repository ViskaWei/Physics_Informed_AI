"""
CNN noise=0.2 三阶段架构搜索 - 配置文件
生成日期: 2025-12-08

使用方法:
1. 复制到 VIT 仓库
2. 运行 Stage A: python cnn_noise02_config.py --stage A
3. 分析 Stage A 结果后，手动填入 STAGE_B_CANDIDATES
4. 运行 Stage B: python cnn_noise02_config.py --stage B
5. 运行 Stage C: python cnn_noise02_config.py --stage C
"""

import itertools
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================
# 基础配置
# ============================================================

DATA_DIR = "/srv/local/tmp/swei20/data/bosz50000/z0/"
NOISE_LEVEL = 0.2
BATCH_SIZE = 2048
LR = 3e-3  # 小 kernel 最优学习率（来自 noise=0.1 经验）

# ============================================================
# Stage A: 小数据粗扫 (32 runs)
# ============================================================

STAGE_A_CONFIG = {
    "train_size": 4000,
    "val_size": 1000,
    "test_size": 1000,
    "epochs": 30,
    "patience": 10,
    "seed": 0,
}

# 搜索空间
STAGE_A_SEARCH_SPACE = {
    "2L": {
        "num_layers": 2,
        "channels": [32, 64],
        "kernels": [5, 7, 9, 11],
        "dilations": [[1, 1], [1, 2]],
    },
    "3L": {
        "num_layers": 3,
        "channels": [32, 64, 64],
        "kernels": [5, 7, 9, 11],
        "dilations": [[1, 1, 1], [1, 2, 4]],
    },
}

WEIGHT_DECAYS = [0, 1e-4]

def generate_stage_a_configs():
    """生成 Stage A 的 32 个配置"""
    configs = []
    
    for layer_type, space in STAGE_A_SEARCH_SPACE.items():
        for k in space["kernels"]:
            for d in space["dilations"]:
                for wd in WEIGHT_DECAYS:
                    d_str = "_".join(map(str, d))
                    wd_str = "wd0" if wd == 0 else "wd1e4"
                    
                    config = {
                        "run_id": f"{layer_type}_k{k}_d{d_str}_{wd_str}",
                        "num_layers": space["num_layers"],
                        "channels": space["channels"],
                        "kernel_size": k,
                        "dilation": d,
                        "weight_decay": wd,
                        **STAGE_A_CONFIG,
                    }
                    configs.append(config)
    
    return configs

# ============================================================
# Stage B: 全训练集精调 (10-16 runs)
# ============================================================

STAGE_B_CONFIG = {
    "train_size": 16000,
    "val_size": 1000,
    "test_size": 1000,
    "epochs": 100,
    "patience": 20,
}

# 🔴 TODO: Stage A 完成后，在这里填入 top 5-8 结构
STAGE_B_CANDIDATES = [
    # 示例格式 (根据 Stage A 结果填入):
    # {"num_layers": 2, "channels": [32, 64], "kernel_size": 9, "dilation": [1, 1], "weight_decay": 0},
    # {"num_layers": 3, "channels": [32, 64, 64], "kernel_size": 9, "dilation": [1, 2, 4], "weight_decay": 0},
]

STAGE_B_SEEDS = [0, 1]

def generate_stage_b_configs():
    """生成 Stage B 的配置（需要先填入 STAGE_B_CANDIDATES）"""
    if not STAGE_B_CANDIDATES:
        print("⚠️ STAGE_B_CANDIDATES 为空！请先运行 Stage A 并填入候选结构。")
        return []
    
    configs = []
    for candidate in STAGE_B_CANDIDATES:
        for seed in STAGE_B_SEEDS:
            d_str = "_".join(map(str, candidate["dilation"]))
            config = {
                "run_id": f"stageB_{candidate['num_layers']}L_k{candidate['kernel_size']}_d{d_str}_seed{seed}",
                "seed": seed,
                **candidate,
                **STAGE_B_CONFIG,
            }
            configs.append(config)
    
    return configs

# ============================================================
# Stage C: 冲上限 (5 runs)
# ============================================================

STAGE_C_CONFIG = {
    "train_size": 16000,  # 可选 32000
    "val_size": 1000,
    "test_size": 1000,
    "epochs": 200,
    "patience": 30,
}

# 🔴 TODO: Stage B 完成后，在这里填入最优结构
STAGE_C_BEST = None
# 示例:
# STAGE_C_BEST = {"num_layers": 3, "channels": [32, 64, 64], "kernel_size": 9, "dilation": [1, 2, 4], "weight_decay": 0}

STAGE_C_SEEDS = [0, 1, 2, 3, 4]

def generate_stage_c_configs():
    """生成 Stage C 的配置（需要先填入 STAGE_C_BEST）"""
    if STAGE_C_BEST is None:
        print("⚠️ STAGE_C_BEST 为空！请先运行 Stage B 并填入最优结构。")
        return []
    
    configs = []
    d_str = "_".join(map(str, STAGE_C_BEST["dilation"]))
    
    for seed in STAGE_C_SEEDS:
        config = {
            "run_id": f"stageC_best_seed{seed}",
            "seed": seed,
            **STAGE_C_BEST,
            **STAGE_C_CONFIG,
        }
        configs.append(config)
    
    return configs

# ============================================================
# 命令生成
# ============================================================

def config_to_command(config: dict) -> str:
    """将配置转换为命令行"""
    channels_str = ",".join(map(str, config["channels"]))
    dilation_str = ",".join(map(str, config["dilation"]))
    
    cmd = f"""python train_cnn.py \\
    --data_dir {DATA_DIR} \\
    --noise {NOISE_LEVEL} \\
    --train_size {config['train_size']} \\
    --val_size {config.get('val_size', 1000)} \\
    --test_size {config.get('test_size', 1000)} \\
    --epochs {config['epochs']} \\
    --patience {config['patience']} \\
    --lr {LR} \\
    --weight_decay {config['weight_decay']} \\
    --batch_size {BATCH_SIZE} \\
    --seed {config['seed']} \\
    --num_layers {config['num_layers']} \\
    --kernel_size {config['kernel_size']} \\
    --dilation {dilation_str} \\
    --channels {channels_str} \\
    --exp_name {config['run_id']}"""
    
    return cmd

def print_stage_commands(stage: str):
    """打印指定 stage 的所有命令"""
    if stage == "A":
        configs = generate_stage_a_configs()
        print(f"=" * 60)
        print(f"Stage A: 小数据粗扫 ({len(configs)} runs)")
        print(f"预计时间: ~{len(configs) * 0.5:.0f} 分钟")
        print(f"=" * 60)
    elif stage == "B":
        configs = generate_stage_b_configs()
        print(f"=" * 60)
        print(f"Stage B: 全训练集精调 ({len(configs)} runs)")
        print(f"预计时间: ~{len(configs) * 3:.0f} 分钟")
        print(f"=" * 60)
    elif stage == "C":
        configs = generate_stage_c_configs()
        print(f"=" * 60)
        print(f"Stage C: 冲上限 ({len(configs)} runs)")
        print(f"预计时间: ~{len(configs) * 5:.0f} 分钟")
        print(f"=" * 60)
    else:
        print(f"未知 stage: {stage}")
        return
    
    for i, config in enumerate(configs, 1):
        print(f"\n# Run {i}/{len(configs)}: {config['run_id']}")
        print(config_to_command(config))

def print_summary_table():
    """打印配置汇总表"""
    configs = generate_stage_a_configs()
    
    print("=" * 80)
    print("Stage A 配置汇总表 (32 runs)")
    print("=" * 80)
    print(f"{'run_id':<25} {'L':>2} {'k':>3} {'dilation':<12} {'wd':>6}")
    print("-" * 80)
    
    for c in configs:
        d_str = str(c["dilation"])
        wd_str = str(c["weight_decay"])
        print(f"{c['run_id']:<25} {c['num_layers']:>2} {c['kernel_size']:>3} {d_str:<12} {wd_str:>6}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CNN noise=0.2 实验配置生成器")
    parser.add_argument("--stage", choices=["A", "B", "C", "summary"], default="summary",
                       help="要生成的 stage (A/B/C) 或 summary")
    
    args = parser.parse_args()
    
    if args.stage == "summary":
        print_summary_table()
    else:
        print_stage_commands(args.stage)
