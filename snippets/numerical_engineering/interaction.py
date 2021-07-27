#https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# creating dummy dataset
X = np.arange(10).reshape(5, 2)
X.shape
>>> (5, 2)

# interactions between features only
interactions = PolynomialFeatures(interaction_only=True)
X_interactions= interactions.fit_transform(X)
X_interactions.shape
>>> (5, 4)

# polynomial features 
polynomial = PolynomialFeatures(5)
X_poly = polynomial.fit_transform(X)
X_poly.shape
>>> (5, 6)