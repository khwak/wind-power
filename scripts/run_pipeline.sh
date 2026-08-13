#!/bin/bash
set -e 

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate wind

# 실행 모드 설정 (chronos, ensemble, chronos_ensemble 중 택 1)
# 'autogluon', 'ensemble', 'ag_ensemble'
# 'final', 'search', 'quantile'
MODE="final"
RUN_OPTIMIZATION=false

echo "=== [1/5] prepare_data.py 시작: $(date) ==="
python prepare_data.py
# python prepare_data_all.py

if [ "$RUN_OPTIMIZATION" = true ]; then
    echo "=== [2/5] 하이퍼파라미터 및 앙상블 최적화 시작: $(date) ==="
    
    echo "--> 1. 손실함수 파라미터 재탐색 (discover_loss_config.py)"
    python discover_loss_config.py
    
    echo "--> 2. 계절/구간별 앙상블 규칙 탐색 (discover_regimes.py)"
    python discover_regimes.py
else
    echo "=== [2/5] 최적화 건너뜀 (기존 config.py 설정 사용) ==="
fi

echo "=== [3/5] train.py 시작 (Mode: $MODE): $(date) ==="
python train.py --mode $MODE

echo "=== [4/5] evaluate.py 시작: $(date) ==="
python evaluate.py

echo "=== [5/5] visualize.py 시작: $(date) ==="
python visualize.py --mode $MODE

duration=$SECONDS
echo "=== 파이프라인 전체 완료: $(date) ==="
echo "=== 총 소요 시간: $(($duration / 60))분 $(($duration % 60))초 ==="