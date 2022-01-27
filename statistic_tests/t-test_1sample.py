from scipy.stats import ttest_1samp
import numpy as np
import pandas as pd

dataset = pd.read_csv()

# Value considered true
truth = 30

var_mean = np.mean(dataset['col1'])
print(var_mean)

tset, pval = ttest_1samp(dataset['col1'], truth)
print("p-value", pval)

if pval < 0.05:
  print("We are rejecting null hypothesis")
else:
  print("We are not rejecting null hypothesis")
