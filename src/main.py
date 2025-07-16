import os
import sys
import subprocess
from pathlib import Path

# 计算项目根目录（src 的上一级）
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# 要依次运行的脚本列表（相对于项目根目录）
scripts = [
    "src/data_preprocessing.py",
    "src/exploratory_analysis.py",
    "src/model_training.py",
    "src/day2_lightgbm_xgboost.py",
    "src/day2_supervised.py",
    "src/day2_single_run_analysis.py"
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running {script} ...")
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"{script} 失败，错误码 {e.returncode}")
        sys.exit(1)

print("\n全部脚本运行完成！")