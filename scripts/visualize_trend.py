import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def plot_yearly_trends():
    print("데이터 로딩 중...")
    data_path = "/home/khwak/wind-power/data/raw/train/train_labels.csv"
    df = pd.read_csv(data_path)
    
    df['kst_dtm'] = pd.to_datetime(df['kst_dtm'])
    df['year'] = df['kst_dtm'].dt.year

    # 2월 29일 제외 (연도별 8760시간으로 길이를 맞추기 위함)
    df = df[~((df['kst_dtm'].dt.month == 2) & (df['kst_dtm'].dt.day == 29))]

    # 연도 오버레이를 위해 모든 연도를 기준 연도(2024)로 통일한 'plot_date' 생성
    df['plot_date'] = df['kst_dtm'].apply(lambda x: x.replace(year=2024))

    TARGET_COLS = ['kpx_group_1', 'kpx_group_2', 'kpx_group_3']

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    for i, target in enumerate(TARGET_COLS):
        print(f"[{target}] 시각화 생성 중...")
        sns.lineplot(
            data=df, 
            x='plot_date', 
            y=target, 
            hue='year', 
            palette='tab10', 
            ax=axes[i],
            alpha=0.7,
            linewidth=0.5
        )
        axes[i].set_title(f"Year-over-Year Power Generation Trend: {target}")
        axes[i].set_ylabel("Power (kWh)")
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # 월(Month) 이름 표시
        
    plt.tight_layout()
    output_path = "/home/khwak/wind-power/outputs/yoy_trend_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ 연도별 시각화 완료! 저장 위치: {output_path}")

if __name__ == "__main__":
    plot_yearly_trends()