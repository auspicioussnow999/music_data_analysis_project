"""
src/day2_supervised.py
用 cluster 标签做有监督分类：LGBM + XGBoost
"""
import os
import warnings
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------- 0. 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# 如果 processed_music_data.csv 没有 cluster，就用聚类结果
proc_path   = ROOT / 'data' / 'processed_music_data.csv'
cluster_csv = ROOT / 'day2_results' / 'clustered_music_data.csv'

if not proc_path.exists() or 'cluster' not in pd.read_csv(proc_path, nrows=0).columns:
    df = pd.read_csv(cluster_csv)
else:
    df = pd.read_csv(proc_path)

# ---------- 1. 数据 ----------
X = df.drop('cluster', axis=1)
y = df['cluster']

# ---------- 2. 交叉验证 ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'lgbm': LGBMClassifier(random_state=42),
    'xgb':  XGBClassifier(random_state=42, use_label_encoder=False)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    pd.DataFrame({'fold': range(1, 6), 'f1_macro': scores}) \
      .to_csv(f'day2_results/{name}_cv.csv', index=False)
    model.fit(X, y)
    joblib.dump(model, f'day2_results/{name}_model.pkl')
    print(f"{name.upper()} 5-fold F1_macro: {scores.round(4)}")

print("🎉 day2_supervised.py 运行完成！")