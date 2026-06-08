#!/bin/bash

echo "action"
module load miniconda3/25.5.1
conda activate ./CONDA/conda_dir/envs/wot

# 定义要处理的 day 文件夹列表
DAYS=("day2" "day4" "day6" "day246")

# 循环处理每一个 day 目录
for DAY in "${DAYS[@]}"; do
    echo "Processing ${DAY} ..."

    TARGET_DIR="./experiment/benchmark/${DAY}"
    DATA_DIR="./data/bench/lt/${DAY}/wot/"

    for i in 1; do
        echo "Run $i for ${DAY}:"
        run_id="${TARGET_DIR}/run${i}"

        if [ ! -d $run_id ]; then
            mkdir -p $run_id
        fi

        echo "doing optimal transport..."
        wot optimal_transport \
            --matrix ${DATA_DIR}hvg.counts.txt \
            --cell_days ${DATA_DIR}days.txt \
            --cell_filter ${DATA_DIR}serum_cell_ids.txt \
            --growth_iters 3 \
            --cell_growth_rates ${DATA_DIR}growth_gs_init.txt \
            --out ${run_id}/res

        echo "calculating fate for HSPC"
        for cell in Monocyte Neutrophil; do
            echo "call ${cell}"
            prefix="${cell}_fate"
            wot fates \
                --tmap ${run_id}/res \
                --cell_set ${DATA_DIR}major_cell_sets.gmt \
                --day 2 \
                --cell_set_filter ${cell} \
                --out ${run_id}/${prefix}
            echo "done"
        done
    done
done