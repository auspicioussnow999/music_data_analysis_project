import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防止显示错误
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, sys
# 把当前脚本所在目录的父目录加入 sys.path 并设为工作目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
# 设置环境变量避免警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 创建保存目录
save_dir = 'notebooks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取处理后的数据
df = pd.read_csv("data/processed_music_data.csv")
original_features = df.columns.tolist()

# 1. 数据标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# 2. 确定最佳聚类数
print("寻找最佳聚类数...")
k_range = range(2, 11)
silhouette_scores = []
inertia = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    inertia.append(kmeans.inertia_)
    print(f"K={k}: 轮廓系数={silhouette_avg:.4f}, 惯性={kmeans.inertia_:.2f}")

# 可视化肘部法则和轮廓系数
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('聚类数 K')
plt.ylabel('惯性 (Inertia)')
plt.title('肘部法则 (Elbow Method)')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('聚类数 K')
plt.ylabel('轮廓系数 (Silhouette Score)')
plt.title('轮廓系数分析')
plt.tight_layout()

# 保存肘部法则和轮廓系数图到 notebooks 文件夹
plt.savefig(os.path.join(save_dir, 'k_selection.png'))
plt.close()

# 选择最佳K值 (轮廓系数最大)
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n最佳聚类数: K={best_k} (轮廓系数={max(silhouette_scores):.4f})")

# 3. 使用最佳K值训练模型
print("\n训练最终聚类模型...")
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(df_scaled)

# 4. 聚类分析
# 计算聚类中心
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=df.columns[:-1])
print("\n聚类中心 (原始尺度):")
print(cluster_centers_df)

# 5. 可视化降维结果
print("\n生成可视化...")
plt.figure(figsize=(15, 5))

# PCA降维
plt.subplot(1, 3, 1)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title(f'PCA 降维 (方差解释率: {sum(pca.explained_variance_ratio_):.2f})')
plt.colorbar(label='Cluster')

# t-SNE降维
plt.subplot(1, 3, 2)
tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE 降维')
plt.colorbar(label='Cluster')

# 特征重要性分析
plt.subplot(1, 3, 3)
feature_importance = pd.DataFrame(kmeans.cluster_centers_, columns=original_features)
feature_importance = feature_importance.abs().mean().sort_values(ascending=False)
feature_importance[:10].plot(kind='barh')  # 显示最重要的10个特征
plt.title('聚类特征重要性')
plt.xlabel('平均绝对中心距离')
plt.tight_layout()

# 保存降维结果图到 notebooks 文件夹
plt.savefig(os.path.join(save_dir, 'cluster_visualization.png'))
plt.close()

# 6. 保存结果
print("\n保存结果...")
df.to_csv(os.path.join(save_dir, "clustered_music_data.csv"), index=False)
cluster_centers_df.to_csv(os.path.join(save_dir, "cluster_centers.csv"), index=False)

# 7. 生成最终报告
final_silhouette = silhouette_score(df_scaled, df["cluster"])
print("\n" + "="*50)
print(f"最终轮廓系数: {final_silhouette:.4f}")
print(f"聚类分布:\n{df['cluster'].value_counts().sort_index()}")
print(f"可视化结果已保存至: {save_dir}")
print("="*50)