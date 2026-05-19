#!/bin/bash
#SBATCH -J DMT.CD8.Run          # 作业名
#SBATCH -A wangjiayi            # 用户名/账户名
#SBATCH -N 1                     # 节点数
#SBATCH -n 1                     # 任务数
#SBATCH -c 16                    # CPU 核数
#SBATCH --gres=gpu:rtx4090:1    # GPU 资源
#SBATCH -o out.%j.log            # 标准输出日志
#SBATCH -e err.%j.log            # 错误日志
#SBATCH --time=2-00:00:00        # 最大运行时间 1天

echo "Starting CD8 configurations run..."

# 加载环境
module load miniconda3/25.5.1
conda activate dymotree

# 进入项目目录
cd /data02/work/wangjiayi/My_project/DyMoTree1.0/

# CD8配置文件列表（排除sample_ratio）
CONFIG_FILES=(
    "Fig2.tune.pretrain.CD8.yaml"
    "Fig2.tune.similarity_mode.CD8.yaml"
)

# 循环执行每个配置文件
for cfg in "${CONFIG_FILES[@]}"
do
    OUTPUT_CSV="./experiment/$(basename "$cfg" .yaml).csv"
    echo "Running config: $cfg -> $OUTPUT_CSV"
    python ./run/run_dymotree.py --config "./config/$cfg" --output_csv "$OUTPUT_CSV"
done

echo "All CD8 configurations finished."
