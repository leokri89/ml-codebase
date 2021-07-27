#https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import pandas as pd

#map the counts to quantiles (adaptive binning)
views_adaptive_bin = pd.qcut(views, 5, labels=False)

print(f"Adaptive bins: {views_adaptive_bin}")
>>> Adaptive bins: [1 3 0 1 4 2 3 4 0 4 0 2 3 1]