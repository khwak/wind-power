#!/bin/bash
set -e 

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate wind

# 실행 모드 설정 (chronos, ensemble, chronos_ensemble 중 택 1)
# 'autogluon', 'ensemble', 'ag_ensemble'
# 'final', 'search'
MODE="search"

echo "=== [1/4] prepare_data.py 시작: $(date) ==="
python prepare_data.py
# python prepare_data_B.py

echo "=== [2/4] train.py 시작 (Mode: $MODE): $(date) ==="
python train.py --mode $MODE
# python train_B.py --mode $MODE

echo "=== [3/4] evaluate.py 시작: $(date) ==="
python evaluate.py

echo "=== [4/4] visualize.py 시작: $(date) ==="
python visualize.py --mode $MODE

duration=$SECONDS
echo "=== 파이프라인 전체 완료: $(date) ==="
echo "=== 총 소요 시간: $(($duration / 60))분 $(($duration % 60))초 ==="