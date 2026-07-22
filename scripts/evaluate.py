import os
import glob
import numpy as np
import pandas as pd
from config import TARGET_COLS, CAPACITY_KWH, OUTPUT_DIR

def metric(answer_df, pred_df):
    """
    실제 발전량이 설비용량의 10% 이상인 시간대만 평가하는 NMAE 및 FICR 산식
    """
    group_nmae = []
    group_ficr = []
    
    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]
        
        # 유효 조건: 실제 발전량이 용량의 10% 이상
        valid = actual >= capacity * 0.10
        actual_v = actual[valid]
        forecast_v = forecast[valid]
        
        if len(actual_v) == 0:
            continue
            
        # NMAE 계산
        error_rate = np.abs(forecast_v - actual_v) / capacity
        group_nmae.append(np.mean(error_rate))
        
        # FICR 계산 (오차율에 따른 정산단가)
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0
        )
        earned_settlement = np.sum(actual_v * unit_price)
        max_settlement = np.sum(actual_v * 4.0)
        
        group_ficr.append(earned_settlement / max_settlement if max_settlement > 0 else 0)
        
    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr
    
    return total_score, one_minus_nmae, ficr


if __name__ == "__main__":
    print("=== 모델 평가(evaluate.py) 시작 ===")
    
    # 1. 정답지(2024년) 데이터 준비 (파일 트리에 맞게 경로 수정)
    train_path = "/home/khwak/wind-power/data/raw/train/train_labels.csv"
    train_df = pd.read_csv(train_path)
    
    # train_labels.csv의 시간 컬럼은 'kst_dtm'임
    train_df['kst_dtm'] = pd.to_datetime(train_df['kst_dtm'])
    
    answer_2024 = train_df[train_df['kst_dtm'].dt.year == 2024].copy()
    
    # 2025년(8760시간)과 길이를 맞추기 위해 2024년 2월 29일(24시간) 제거
    answer_2024 = answer_2024[~((answer_2024['kst_dtm'].dt.month == 2) & (answer_2024['kst_dtm'].dt.day == 29))]
    answer_2024 = answer_2024.sort_values('kst_dtm').reset_index(drop=True)
    
    print(f"✅ 2024년 Proxy 정답지 세팅 완료 (총 {len(answer_2024)}시간)")

    # 2. outputs 폴더 내 모든 submission 파일 탐색 (하위 폴더 recursive 탐색)
    submission_files = glob.glob(str(OUTPUT_DIR / "**" / "submission_*.csv"), recursive=True)
    
    if not submission_files:
        print(f"❌ 평가할 submission 파일이 {OUTPUT_DIR} 폴더 하위에 없습니다.")
    else:
        print(f"🔎 총 {len(submission_files)}개의 모델 결과를 평가합니다.\n" + "-"*60)
        
        for file_path in sorted(submission_files):
            file_name = os.path.basename(file_path)
            pred_df = pd.read_csv(file_path)
            
            # 3. 데이터 검증 및 평가 수행
            if len(pred_df) != len(answer_2024):
                print(f"⚠️ [Skip] {file_name}: 데이터 길이 불일치 (예측 {len(pred_df)} vs 정답 {len(answer_2024)})")
                continue
                
            pred_df['forecast_kst_dtm'] = pd.to_datetime(pred_df['forecast_kst_dtm'])
            pred_df = pred_df.sort_values('forecast_kst_dtm').reset_index(drop=True)
            
            # 산식 적용
            total_score, nmae_score, ficr_score = metric(answer_2024, pred_df)
            
            # 결과 출력
            print(f"📊 {file_name}")
            print(f"   Total Score : {total_score:.4f} | (1 - NMAE) : {nmae_score:.4f} | FICR : {ficr_score:.4f}")
        print("-" * 60)