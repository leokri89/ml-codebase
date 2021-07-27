#https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import numpy as np 

#15 random integers from the "discrete uniform" distribution
ages = np.random.randint(0, 100, 15)

#evenly spaced bins
ages_binned = np.floor_divide(ages, 10)

print(f"Ages: {ages} \nAges Binned: {ages_binned} \n")
>>> Ages: [97 56 43 73 89 68 67 15 18 36  4 97 72 20 35]
Ages Binned: [9 5 4 7 8 6 6 1 1 3 0 9 7 2 3]

#numbers spanning several magnitudes
views = [300, 5936, 2, 350, 10000, 743, 2854, 9113, 25, 20000, 160, 683, 7245, 224]

#map count -> exponential width bins
views_exponential_bins = np.floor(np.log10(views))

print(f"Views: {views} \nViews Binned: {views_exponential_bins}")
>>> Views: [300, 5936, 2, 350, 10000, 743, 2854, 9113, 25, 20000, 160, 683, 7245, 224]
Views Binned: [2. 3. 0. 2. 4. 2. 3. 3. 1. 4. 2. 2. 3. 2.]