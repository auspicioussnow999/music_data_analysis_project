import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
# 把当前脚本所在目录的父目录加入 sys.path 并设为工作目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
# 读取处理后的数据
df = pd.read_csv("data/processed_music_data.csv")

# 绘制数据的分布图
df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.savefig("notebooks/data_distribution.png")
plt.close()

# 绘制相关性热图
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("notebooks/correlation_heatmap.png")
plt.close()