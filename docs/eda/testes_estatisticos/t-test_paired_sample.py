import pandas as pd
from scipy import stats

df = pd.read_csv("blood_pressure.csv")

df[['bp_before','bp_after']].describe()

ttest,pval = stats.ttest_rel(df['bp_before'], df['bp_after'])
print(pval)

if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")