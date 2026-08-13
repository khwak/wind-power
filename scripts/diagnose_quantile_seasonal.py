import os
import sys
from pathlib import Path
from config import TARGET_COLS, CAPACITY_KWH, SEASONAL_CV_FOLD_STARTS
from train import diagnose_quantile_calibration_seasonal

if __name__ == "__main__":
    output_dir = Path("/home/khwak/wind-power/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(output_dir)

    # txt 대신 log 확장자로 변경
    log_file_path = output_dir / "calibration_results.log"
    with open(log_file_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            for target in TARGET_COLS:
                print(f"\n{'='*60}\n{target} 다중fold quantile calibration 진단\n{'='*60}")
                diagnose_quantile_calibration_seasonal(
                    target, CAPACITY_KWH[target], SEASONAL_CV_FOLD_STARTS[target]
                )
        finally:
            sys.stdout = original_stdout

    print(f"✅ 진단 완료! 모든 로그 결과와 그래프가 {output_dir} 폴더에 저장되었습니다.")