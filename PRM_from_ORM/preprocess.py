import json
from datasets import load_dataset

import pandas as pd

import math
from collections import Counter
import pandas as pd


dataset = load_dataset(
  "json",
  data_files={
      "gsm8k": "./PRM_from_ORM/ProcessBench/gsm8k.json",
      "math": "./PRM_from_ORM/ProcessBench/math.json",
      "olympiadbench": "./PRM_from_ORM/ProcessBench/olympiadbench.json",
      "omnimath": "./PRM_from_ORM/ProcessBench/omnimath.json",
  }
)


scan_pro = pd.read_parquet('./PRM_from_ORM/scan_pro.parquet')


print('da')