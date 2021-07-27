#https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import plotly.graph_objects as go

wine_json= load_wine() # load in dataset

df = pd.DataFrame(data=wine_json["data"], columns=wine_json["feature_names"]) # create pandas dataframe

df["Target"] = wine_json["target"] # created new column and added target labels

# standardization
std_scaler = StandardScaler().fit(df[["alcohol", "malic_acid"]])
df_std = std_scaler.transform(df[["alcohol", "malic_acid"]])

# minmax scaling
minmax_scaler = MinMaxScaler().fit(df[["alcohol", "malic_acid"]])
df_minmax = minmax_scaler.transform(df[["alcohol", "malic_acid"]])

# l2 normalization
l2norm = Normalizer().fit(df[["alcohol", "malic_acid"]])
df_l2norm = l2norm.transform(df[["alcohol", "malic_acid"]])

# creating traces
trace1 = go.Scatter(x= df_std[:, 0],
                    y= df_std[:, 1],
                    mode= "markers",
                    name= "Standardized Scale")

trace2 = go.Scatter(x= df_minmax[:, 0],
                    y= df_minmax[:, 1],
                    mode= "markers",
                    name= "MinMax Scale")

trace3 = go.Scatter(x= df_l2norm[:, 0],
                    y= df_l2norm[:, 1],
                    mode= "markers",
                    name= "L2 Norm Scale")

trace4 = go.Scatter(x= df["alcohol"],
                    y= df["malic_acid"],
                    mode= "markers",
                    name= "Original Scale")

layout = go.Layout(
         title= "Effects of Feature scaling",
         xaxis=dict(title= "Alcohol"),
         yaxis=dict(title= "Malic Acid")
         )

data = [trace1, trace2, trace3, trace4]
fig = go.Figure(data=data, layout=layout)
fig.show()