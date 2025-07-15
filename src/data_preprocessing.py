import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os, sys
# 把当前脚本所在目录的父目录加入 sys.path 并设为工作目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
# 读取数据
df = pd.read_csv("data/nigerian-songs.csv")

# 处理缺失值
imputer = SimpleImputer(strategy="mean")  # 使用均值填充
df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['number'])), columns=df.select_dtypes(include=['number']).columns)

# 标准化数值特征
scaler = StandardScaler()
df_imputed[df.select_dtypes(include=['number']).columns] = scaler.fit_transform(df_imputed)

# 保存处理后的数据
df_imputed.to_csv("data/processed_music_data.csv", index=False)
print('已保存')