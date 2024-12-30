#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --time=23:59:59
#SBATCH --job-name=rlds
#SBATCH --output=logs/rlds-%J.log
#SBATCH --error=logs/rlds-%J.err

# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files
# in /tmp to store dataset during conversion
ulimit -n 20000

# TODO: MODIFY data_dir and target_dir
uv run python scripts/data/modify_rlds_dataset.py \
    --dataset=bridge \
    --data_dir=/n/fs/llm-unc/data/ \
    --target_dir=/n/fs/llm-unc/data/resize_224 \
    --mods=resize_and_jpeg_encode \
    --n_workers=40 \
    --max_episodes_in_memory=200
