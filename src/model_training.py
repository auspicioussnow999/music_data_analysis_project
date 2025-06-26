import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 读取处理后的数据
df = pd.read_csv("../data/processed_music_data.csv")

# 训练 K-Means 聚类模型
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(df)

# 计算聚类效果的轮廓系数
silhouette_avg = silhouette_score(df.drop("cluster", axis=1), df["cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# 保存聚类结果
df.to_csv("../data/clustered_music_data.csv", index=False)