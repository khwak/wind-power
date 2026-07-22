import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path

# 한글 폰트 및 마이너스 깨짐 설정 (Linux 환경)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def visualize_submission(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    df = pd.read_csv(file_path)
    df['forecast_kst_dtm'] = pd.to_datetime(df['forecast_kst_dtm'])
    df.set_index('forecast_kst_dtm', inplace=True)
    
    target_cols = ['kpx_group_1', 'kpx_group_2', 'kpx_group_3']
    
    # 1. 전체 1년 예측 시계열 플롯
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(f"2025 Full-Year Forecast: {file_path.name}", fontsize=14)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, col in enumerate(target_cols):
        axes[i].plot(df.index, df[col], label=col, color=colors[i], alpha=0.8, linewidth=0.8)
        axes[i].set_ylabel("Power (kWh)")
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
        
    plt.xlabel("Date")
    plt.tight_layout()
    
    save_path_ts = file_path.parent / f"{file_path.stem}_timeseries.png"
    plt.savefig(save_path_ts, dpi=200)
    plt.close()
    
    # 2. 첫 1개월(1월) 상세 시계열 플롯 (초기 패턴 확인용)
    df_jan = df.loc['2025-01-01':'2025-01-31']
    fig, ax = plt.subplots(figsize=(15, 4))
    for i, col in enumerate(target_cols):
        ax.plot(df_jan.index, df_jan[col], label=col, color=colors[i], alpha=0.8)
    ax.set_title(f"January 2025 Detailed Forecast (First Month)")
    ax.set_ylabel("Power (kWh)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path_jan = file_path.parent / f"{file_path.stem}_january.png"
    plt.savefig(save_path_jan, dpi=200)
    plt.close()

    print(f"✅ 시각화 완료! 저장된 이미지:")
    print(f"   1. {save_path_ts}")
    print(f"   2. {save_path_jan}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="chronos", help="Submission mode name")
    args = parser.parse_args()
    
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    
    # 해당 모드(mode)의 가장 최근 생성된 submission 파일 검색
    pattern = str(output_dir / f"submission_{args.mode}_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        # 타임스탬프가 없는 기존 형식도 예외 처리
        fallback_file = output_dir / f"submission_{args.mode}.csv"
        if fallback_file.exists():
            files = [str(fallback_file)]
        else:
            print(f"❌ '{args.mode}' 모드의 제출 파일을 찾을 수 없습니다.")
            exit(1)
            
    # 가장 최근에 생성/수정된 파일 선택
    latest_file = max(files, key=Path)
    print(f"시각화 대상 최신 파일: {latest_file}")
    
    visualize_submission(latest_file)