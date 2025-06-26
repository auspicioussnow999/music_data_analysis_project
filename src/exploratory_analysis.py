import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取处理后的数据
df = pd.read_csv("../data/processed_music_data.csv")

# 绘制数据的分布图
df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.savefig("../notebooks/data_distribution.png")
plt.close()

# 绘制相关性热图
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("../notebooks/correlation_heatmap.png")
plt.close()