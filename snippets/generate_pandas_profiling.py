TRAIN_SET = "./datasets/train.csv"
TEST_SET = "./datasets/test.csv"

PROFILING_CONFIG = "./profiling_config.yml"

import pandas as pd
import pandas_profiling as pdp

df = pd.read_csv(TRAIN_SET)

profile = pdp.ProfileReport(df, config_file=PROFILING_CONFIG)
profile.to_file("profiling.html")
