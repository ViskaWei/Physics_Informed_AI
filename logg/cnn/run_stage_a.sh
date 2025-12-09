#!/bin/bash
# ============================================================
# CNN noise=0.2 Stage A: 小数据粗扫 (32 runs)
# 生成日期: 2025-12-08
# 预计时间: ~30 分钟
# ============================================================

set -e  # 遇到错误停止

# 基础配置
DATA_DIR="/srv/local/tmp/swei20/data/bosz50000/z0/"
NOISE=0.2
TRAIN_SIZE=4000
EPOCHS=30
PATIENCE=10
LR=3e-3
BATCH_SIZE=2048
SEED=0

# 结果目录
RESULTS_DIR="./results/noise02_stage_a"
mkdir -p $RESULTS_DIR

# 计数器
total=32
current=0

echo "============================================================"
echo "CNN noise=0.2 Stage A: 小数据粗扫"
echo "总计: $total runs"
echo "预计时间: ~30 分钟"
echo "============================================================"

# 2层 CNN 配置
for k in 5 7 9 11; do
    for d in "1,1" "1,2"; do
        for wd in 0 1e-4; do
            current=$((current + 1))
            d_name=$(echo $d | tr ',' '_')
            wd_name=$([ "$wd" = "0" ] && echo "wd0" || echo "wd1e4")
            run_id="2L_k${k}_d${d_name}_${wd_name}"
            
            echo ""
            echo "[$current/$total] Running: $run_id"
            echo "  kernel=$k, dilation=[$d], wd=$wd"
            
            python train_cnn.py \
                --data_dir $DATA_DIR \
                --noise $NOISE \
                --train_size $TRAIN_SIZE \
                --epochs $EPOCHS \
                --patience $PATIENCE \
                --lr $LR \
                --weight_decay $wd \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --num_layers 2 \
                --kernel_size $k \
                --dilation $d \
                --channels "32,64" \
                --exp_name $run_id \
                2>&1 | tee "$RESULTS_DIR/${run_id}.log"
        done
    done
done

# 3层 CNN 配置
for k in 5 7 9 11; do
    for d in "1,1,1" "1,2,4"; do
        for wd in 0 1e-4; do
            current=$((current + 1))
            d_name=$(echo $d | tr ',' '_')
            wd_name=$([ "$wd" = "0" ] && echo "wd0" || echo "wd1e4")
            run_id="3L_k${k}_d${d_name}_${wd_name}"
            
            echo ""
            echo "[$current/$total] Running: $run_id"
            echo "  kernel=$k, dilation=[$d], wd=$wd"
            
            python train_cnn.py \
                --data_dir $DATA_DIR \
                --noise $NOISE \
                --train_size $TRAIN_SIZE \
                --epochs $EPOCHS \
                --patience $PATIENCE \
                --lr $LR \
                --weight_decay $wd \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --num_layers 3 \
                --kernel_size $k \
                --dilation $d \
                --channels "32,64,64" \
                --exp_name $run_id \
                2>&1 | tee "$RESULTS_DIR/${run_id}.log"
        done
    done
done

echo ""
echo "============================================================"
echo "Stage A 完成！"
echo "日志保存在: $RESULTS_DIR/"
echo ""
echo "下一步:"
echo "1. 分析结果，选出 top 5-8 结构"
echo "2. 更新 cnn_noise02_config.py 中的 STAGE_B_CANDIDATES"
echo "3. 运行 Stage B"
echo "============================================================"
