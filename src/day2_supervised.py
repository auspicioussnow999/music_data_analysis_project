"""
src/day2_supervised.py
终极集成：全部特征 + log1p + 单线程网格 + 融合 + 还原 MAE
"""
import os, warnings, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from textblob import TextBlob   # pip install textblob

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
SAVE_DIR = 'day2_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 1. 读数据 ----------
df = pd.read_csv('data/nigerian-songs.csv')

# ---------- 2. 构造全部特征 ----------
num_cols = ['danceability','acousticness','energy','instrumentalness',
            'liveness','loudness','speechiness','tempo','time_signature','length']

# 艺人行为
df['artist_avg_pop']  = df.groupby('artist')['popularity'].transform('mean')
df['artist_song_cnt'] = df.groupby('artist')['popularity'].transform('count')
df['artist_pop_std']  = df.groupby('artist')['popularity'].transform('std')

# 时间/顺序
df = df.sort_values('release_date')
df['release_order'] = range(len(df))

# 交叉
df['energy_loud']  = df['energy'] * df['loudness']
df['dance_energy'] = df['danceability'] * df['energy'] / (df['acousticness'] + 1e-5)

# 合作数
df['collab_cnt'] = df['artist'].str.count('feat\\.|&') + 1

# 歌词情感（缺失则 0）
if 'lyrics' in df.columns:
    df['lyrics_sentiment'] = df['lyrics'].fillna('').apply(
        lambda x: TextBlob(str(x)).sentiment.polarity)
else:
    df['lyrics_sentiment'] = 0.0

# ---------- 3. 目标变换 ----------
y_raw = df['popularity']
y = np.log1p(y_raw)          # log1p 变换
feat_cols = num_cols + ['artist_avg_pop','artist_song_cnt','artist_pop_std',
                        'release_order','energy_loud','dance_energy',
                        'collab_cnt','lyrics_sentiment']
X = df[feat_cols].fillna(0)

# ---------- 4. 交叉验证 ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------- 5. 单线程轻量网格 ----------
lgbm = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    random_state=42,
    verbosity=-1          # 关闭 LightGBM 提示
)
xgb = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42,
    verbosity=0
)

# ---------- 6. 交叉验证打印原始 MAE ----------
def cv_mae_raw(model):
    maes = []
    for tr, va in cv.split(X):
        model.fit(X.iloc[tr], np.log1p(y_raw.iloc[tr]))
        pred_raw = np.expm1(model.predict(X.iloc[va]))
        maes.append(mean_absolute_error(y_raw.iloc[va], pred_raw))
    return np.array(maes)

lgb_maes = cv_mae_raw(lgbm)
xgb_maes = cv_mae_raw(xgb)
print("LGBM 5-fold MAE (原始):", lgb_maes.round(4))
print("XGB  5-fold MAE (原始):", xgb_maes.round(4))

# ---------- 7. 融合 ----------
ensemble = VotingRegressor([('lgb', lgbm), ('xgb', xgb)])
ensemble_maes = cv_mae_raw(ensemble)
print("融合 5-fold MAE (原始):", ensemble_maes.round(4))
print("融合平均 MAE (原始):", ensemble_maes.mean().round(4))

# ---------- 8. 保存 ----------
ensemble.fit(X, y)
joblib.dump(ensemble, f'{SAVE_DIR}/ensemble_final.pkl')
print("✅ 全部完成！")