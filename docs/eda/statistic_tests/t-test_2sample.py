from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

sample1 = pd.read_csv(sample1)
sample2 = pd.read_csv(sample2)

sample1_mean = np.mean(sample1['col1'])
sample2_mean = np.mean(sample2['col1'])

sample1_std = np.std(sample1['col1'])
sample2_std = np.std(sample2['col1'])

ttest, pval = ttest_ind(sample1['col1'], sample2['col1'])

print("sample1 mean value:",sample1_mean)
print("sample2 mean value:",sample2_mean)

print("sample1 std value:",sample1_std)
print("sample2 std value:",sample2_std)

print("p-value",pval)

if pval < 0.05:
  print("we are rejeting null hypothesis")
else:
  print("we are not rejecting null hypothesis")
