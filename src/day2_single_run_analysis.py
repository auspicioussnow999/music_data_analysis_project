# src/day2_single_run_analysis.py
import os, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from textblob import TextBlob

# 设置路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
SAVE_DIR = 'day2_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取数据
df = pd.read_csv('data/nigerian-songs.csv')

# 构造全部特征
num_cols = ['danceability','acousticness','energy','instrumentalness',
            'liveness','loudness','speechiness','tempo','time_signature','length']

df['artist_avg_pop']  = df.groupby('artist')['popularity'].transform('mean')
df['artist_song_cnt'] = df.groupby('artist')['popularity'].transform('count')
df['artist_pop_std']  = df.groupby('artist')['popularity'].transform('std')
df = df.sort_values('release_date')
df['release_order'] = range(len(df))
df['energy_loud']  = df['energy'] * df['loudness']
df['dance_energy'] = df['danceability'] * df['energy'] / (df['acousticness'] + 1e-5)
df['collab_cnt'] = df['artist'].str.count('feat\\.|&') + 1
if 'lyrics' in df.columns:
    df['lyrics_sentiment'] = df['lyrics'].fillna('').apply(
        lambda x: TextBlob(str(x)).sentiment.polarity)
else:
    df['lyrics_sentiment'] = 0.0

# 目标变换
y_raw = df['popularity']
y = np.log1p(y_raw)

# 特征全集
feat_all = num_cols + ['artist_avg_pop','artist_song_cnt','artist_pop_std',
                       'release_order','energy_loud','dance_energy',
                       'collab_cnt','lyrics_sentiment']

# 特征精选
feat_select = ['energy','loudness','artist_avg_pop','dance_energy','release_order']

# 训练函数
def train_and_analyze(X, y, y_raw, label):
    X = X.fillna(0)
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=42)

    models = {
        'LGBM': LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1),
        'XGB': XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        mae = mean_absolute_error(y_raw_test, pred)
        results.append((label, name, mae))

        # 错误样本分析
        df_test = pd.DataFrame({
            'actual': y_raw_test.values,
            'pred': pred,
            'error': np.abs(y_raw_test.values - pred)
        })
        worst = df_test.sort_values('error', ascending=False).head(2)
        best = df_test.sort_values('error').head(2)

        print(f"\n[{label} - {name}] MAE: {mae:.4f}")
        print("最差样本（错误大）:")
        print(worst)
        print("最好样本（错误小）:")
        print(best)

    return results

# 实验对比
print("\n=== 全部特征 ===")
res_all = train_and_analyze(df[feat_all], y, y_raw, "全部特征")

print("\n=== 精选特征 ===")
res_select = train_and_analyze(df[feat_select], y, y_raw, "精选特征")

# 保存结果
pd.DataFrame(res_all + res_select, columns=['FeatureSet','Model','MAE']).to_csv(
    f"{SAVE_DIR}/experiment_results.csv", index=False)
print("\n单轮实验完成，结果已保存至 day2_results/experiment_results.csv")