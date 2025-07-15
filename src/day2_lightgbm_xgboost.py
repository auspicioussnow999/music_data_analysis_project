import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 0. 路径 & 目录
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
SAVE_DIR = 'day2_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------
# 1. 读取数据
# -------------------------------------------------
df = pd.read_csv('data/processed_music_data.csv')
X = df.values            # 纯数值矩阵
feature_names = df.columns.tolist()

# -------------------------------------------------
# 2. 标准化
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 3. 选最佳 K（2~10）
# -------------------------------------------------
k_range = range(2, 11)
sil_scores = []
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
    inertias.append(km.inertia_)

# 画图：肘部法则 + 轮廓系数
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o')
plt.title('肘部法则 (Inertia)')
plt.xlabel('K')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(k_range, sil_scores, marker='o', color='r')
plt.title('轮廓系数')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'k_selection.png'))
plt.close()

best_k = k_range[np.argmax(sil_scores)]
print(f"最佳聚类数 K = {best_k} (Silhouette={max(sil_scores):.4f})")

# -------------------------------------------------
# 4. 用最佳 K 训练最终模型
# -------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 保存结果
df_out = df.copy()
df_out['cluster'] = labels
df_out.to_csv(os.path.join(SAVE_DIR, 'clustered_music_data.csv'), index=False)

centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=feature_names
)
centers.to_csv(os.path.join(SAVE_DIR, 'cluster_centers.csv'), index=False)
joblib.dump(kmeans, os.path.join(SAVE_DIR, 'kmeans_model.pkl'))

# -------------------------------------------------
# 5. 可视化（PCA + t-SNE）
# -------------------------------------------------
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42)

df_pca = pca.fit_transform(X_scaled)
df_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', s=8)
plt.title('PCA 2D')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=labels, cmap='viridis', s=8)
plt.title('t-SNE 2D')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'cluster_visualization.png'))
plt.close()

# -------------------------------------------------
# 6. 打印总结
# -------------------------------------------------
print("\n" + "="*50)
print(f"最终轮廓系数: {silhouette_score(X_scaled, labels):.4f}")
print("聚类分布:")
print(df_out['cluster'].value_counts().sort_index())
print("结果已保存到:", SAVE_DIR)
print("="*50)