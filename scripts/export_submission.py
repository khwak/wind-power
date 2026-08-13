import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Config 및 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent if "scripts" in str(Path(__file__)) else Path(__file__).resolve().parent
SAMPLE_SUB_PATH = BASE_DIR / "data" / "raw" / "sample_submission.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FINAL_SUB_PATH = OUTPUT_DIR / "final_submission.csv"

# 설비용량 상한선 (kWh)
CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0
}
TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

def create_and_verify_submission(input_pred_path: str = None):
    print("=== [제출 파일 생성 및 무결성 검증 프로세스] ===")
    
    # 1. sample_submission.csv 로드 (기준 틀)
    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(f"❌ sample_submission.csv 파일을 찾을 수 없습니다: {SAMPLE_SUB_PATH}")
    
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    print(f"✅ sample_submission.csv 로드 완료 (행 수: {len(sample_sub)})")

    # 2. 가장 최근 생성된 submission 파일 자동 탐색 (입력 경로가 없을 경우)
    if input_pred_path is None:
        sub_files = sorted(OUTPUT_DIR.glob("**/*.csv"))
        sub_files = [f for f in sub_files if "final_submission" not in f.name]
        if not sub_files:
            raise FileNotFoundError(f"❌ {OUTPUT_DIR} 내에 예측 결과 CSV 파일이 없습니다.")
        input_pred_path = sub_files[-1]
        
    print(f"📂 대상 예측 파일: {input_pred_path}")
    pred_df = pd.read_csv(input_pred_path)

    # 3. 데이터 검증 및 매칭
    # 3-1. 행 수 검증
    assert len(pred_df) == len(sample_sub), f"❌ 행 수 불일치! (예측: {len(pred_df)}행 vs 양식: {len(sample_sub)}행)"
    
    # 3-2. final_sub 객체 생성 (forecast_id, forecast_kst_dtm 원본 고정)
    final_sub = sample_sub[["forecast_id", "forecast_kst_dtm"]].copy()

    # 3-3. 예측값 채우기 및 클리핑 (0 미만 방지, 설비용량 초과 방지)
    for col in TARGET_COLS:
        if col not in pred_df.columns:
            raise KeyError(f"❌ 예측 파일에 '{col}' 컬럼이 존재하지 않습니다.")
        
        preds = pred_df[col].values
        
        # NaN 체크
        if np.isnan(preds).any():
            nan_cnt = np.isnan(preds).sum()
            raise ValueError(f"❌ '{col}' 예측값 중 {nan_cnt}개의 NaN(결측치)이 발견되었습니다!")

        # Range Clipping (0 ~ CAPACITY_KWH)
        clipped_preds = np.clip(preds, 0.0, CAPACITY_KWH[col])
        final_sub[col] = clipped_preds

    # 4. 키 일치 여부 최종 검증
    assert (final_sub["forecast_id"] == sample_sub["forecast_id"]).all(), "❌ forecast_id 순서/값이 sample_submission과 일치하지 않습니다."
    assert (final_sub["forecast_kst_dtm"] == sample_sub["forecast_kst_dtm"]).all(), "❌ forecast_kst_dtm 순서/값이 sample_submission과 일치하지 않습니다."

    # 5. 대회 제출 규격(utf-8-sig 인코딩, index=False)으로 저장
    final_sub.to_csv(FINAL_SUB_PATH, index=False, encoding="utf-8-sig")
    
    print("-" * 60)
    print(f"최종 제출 파일 생성 완벽 성공!")
    print(f"저장 위치: {FINAL_SUB_PATH}")
    print(f"최종 규격 점검:")
    print(f"   • 행/열 크기 : {final_sub.shape} (8760행, 5열)")
    print(f"   • 컬럼 구성   : {list(final_sub.columns)}")
    print(f"   • 인코딩     : UTF-8 with BOM (utf-8-sig)")
    print(f"   • 결측치 수   : {final_sub.isnull().sum().sum()}개")
    print("-" * 60)
    print(final_sub.head(3))

if __name__ == "__main__":
    # 특정 파일 지정시: create_and_verify_submission("/path/to/your/pred.csv")
    create_and_verify_submission("/home/khwak/wind-power/outputs/submission_final_20260813_111104.csv")