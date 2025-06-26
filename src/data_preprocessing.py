import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv("../data/nigerian-songs.csv")

# 处理缺失值
imputer = SimpleImputer(strategy="mean")  # 使用均值填充
df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['number'])), columns=df.select_dtypes(include=['number']).columns)

# 标准化数值特征
scaler = StandardScaler()
df_imputed[df.select_dtypes(include=['number']).columns] = scaler.fit_transform(df_imputed)

# 保存处理后的数据
df_imputed.to_csv("../data/processed_music_data.csv", index=False)
