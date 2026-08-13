import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

# config.py 경로 가설
try:
    from config import RAW_DIR, OUTPUT_DIR, MODEL_DIR, TARGET_COLS, GROUP_META
except ImportError:
    # 로컬 테스트용 폴백
    RAW_DIR = Path("/home/khwak/wind-power/data/raw")
    OUTPUT_DIR = Path("/home/khwak/wind-power/outputs")
    TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
    GROUP_META = {
        1: {"maker": "vestas", "turbines": list(range(1, 7))},
        2: {"maker": "vestas", "turbines": list(range(7, 13))},
        3: {"maker": "unison", "turbines": list(range(1, 6))},
    }

def _turbine_names(maker: str, turbine_numbers: list[int]) -> list[str]:
    return [f"{maker}_wtg{i:02d}" for i in turbine_numbers]

def extract_monthly_wind_profiles(scada_df, ws_cols):
    """
    SCADA 실측 풍속 데이터를 바탕으로 각 월(Month)을 대표하는 다차원 특징 벡터를 추출합니다.
    - 시간대별(0~23시) 평균 풍속
    - 시간대별(0~23시) 풍속의 표준편차 (변동성)
    """
    # 1. 터빈별 풍속을 평균내어 단지(그룹) 대표 풍속 계산
    scada_df['group_ws_mean'] = scada_df[ws_cols].mean(axis=1)
    
    # 2. 월, 시간 정보 추출
    scada_df['month'] = scada_df['kst_dtm'].dt.month
    scada_df['hour'] = scada_df['kst_dtm'].dt.hour
    
    # 3. 월별/시간대별 통계 집계
    profile_df = scada_df.groupby(['month', 'hour'])['group_ws_mean'].agg(['mean', 'std']).reset_index()
    
    # 결측치 처리 (만약 데이터가 빈 곳이 있다면)
    profile_df = profile_df.bfill().ffill()
    
    profiles = []
    # 1월부터 12월까지 순서대로 벡터 구성
    for m in range(1, 13):
        month_data = profile_df[profile_df['month'] == m].sort_values('hour')
        
        # 24개 시간대의 평균과 표준편차를 이어 붙여 48차원 벡터 생성
        mean_vec = month_data['mean'].to_numpy()
        std_vec = month_data['std'].to_numpy()
        
        # 만약 특정 달의 데이터가 누락되었다면 0으로 채움 방지 (예외 처리)
        if len(mean_vec) != 24:
            raise ValueError(f"{m}월의 시간대별 데이터가 불완전합니다.")
            
        feature_vector = np.concatenate([mean_vec, std_vec])
        profiles.append(feature_vector)
        
    return np.array(profiles) # (12, 48) 행렬 반환

def cluster_months(profiles, num_clusters=4):
    """
    월별 특징 벡터를 계층적 군집화(Hierarchical Clustering)를 통해 묶습니다.
    """
    # 상관관계 기반 거리 계산 (1 - Pearson Correlation)
    # 풍속의 절대적 크기(스케일)보다 오르고 내리는 '패턴'의 유사성에 집중합니다.
    dist_matrix = pdist(profiles, metric='correlation')
    
    # Ward 연결법: 군집 내 분산 증가량을 최소화하는 방식
    Z = linkage(dist_matrix, method='ward')
    
    # 지정한 군집 개수(num_clusters)로 분할
    labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    
    # 결과 포맷팅: {Cluster ID: [월 리스트]}
    clusters = {}
    for i, label in enumerate(labels):
        month = i + 1
        label_key = f"season_{int(label)}"  # numpy int32를 기본 int로 바꾼 후 문자열 조합
        if label_key not in clusters:
            clusters[label_key] = []
        clusters[label_key].append(month)
        
    # 출력 순서 정렬
    return {k: sorted(v) for k, v in clusters.items()}

def main():
    print("="*60)
    print("SCADA 관측 풍속 기반 월별 계절 군집화 시작")
    print("="*60)
    
    # 데이터 로드
    vestas_path = RAW_DIR / "train" / "scada_vestas_train.csv"
    unison_path = RAW_DIR / "train" / "scada_unison_train.csv"
    
    scada_vestas = pd.read_csv(vestas_path, parse_dates=['kst_dtm'])
    scada_unison = pd.read_csv(unison_path, parse_dates=['kst_dtm'])
    
    # 노이즈/결측치 필터링 (발전기 가동 중지 등 비정상 상태 제외)
    # 0.5 미만의 너무 낮은 풍속은 센서 오류나 정지 상태일 수 있으므로 분석에서 제외
    for col in scada_vestas.columns:
        if '_ws' in col:
            scada_vestas.loc[scada_vestas[col] < 0.5, col] = np.nan
    for col in scada_unison.columns:
        if '_ws' in col:
            scada_unison.loc[scada_unison[col] < 0.5, col] = np.nan

    all_clusters = {}
    
    for group, meta in GROUP_META.items():
        print(f"\n[KPX Group {group}] 분석 중...")
        
        # 1. 사용할 데이터 소스 및 컬럼 식별
        df = scada_vestas if meta["maker"] == "vestas" else scada_unison
        turbines = _turbine_names(meta["maker"], meta["turbines"])
        ws_cols = [f"{t}_ws" for t in turbines]
        
        # 2. 특징 벡터 추출 (12개월 x 48차원)
        try:
            profiles = extract_monthly_wind_profiles(df, ws_cols)
        except Exception as e:
            print(f"특징 추출 실패: {e}")
            continue
            
        # 3. 군집화 수행 (4계절을 대체할 4개의 시즌으로 분할)
        clusters = cluster_months(profiles, num_clusters=4)
        
        # 4. 결과 출력
        print(f"군집화 결과:")
        for cluster_id, months in clusters.items():
            print(f"  시즌 {cluster_id}: {months}월")
            
        all_clusters[f"kpx_group_{group}"] = clusters
        
    # 결과 저장 
    output_path = MODEL_DIR / "seasonal_clusters.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_clusters, f, indent=2)
    print(f"\n✅ 군집화 결과 저장 완료: {output_path}")

if __name__ == "__main__":
    main()