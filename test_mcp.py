import os
import pandas as pd

# 替换路径（$HOME 会自动展开）
path = os.path.expandvars("$HOME/data/gsm8k/train.parquet")

# 加载 parquet 数据
df = pd.read_parquet(path)



print('da')