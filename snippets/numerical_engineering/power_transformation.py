# https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

df = pd.read_csv("../data/raw/train.csv")

# applying various transformations
x_log = np.log(df["GrLivArea"].copy())  # log
x_square_root = np.sqrt(
    df["GrLivArea"].copy()
)  # square root x_boxcox, _ = stats.boxcox(df["GrLivArea"].copy()) # boxcox
x = df["GrLivArea"].copy()  # original data

# creating the figures
fig = make_subplots(
    rows=2,
    cols=2,
    horizontal_spacing=0.125,
    vertical_spacing=0.125,
    subplot_titles=(
        "Original Data",
        "Log Transformation",
        "Square root transformation",
        "Boxcox Transformation",
    ),
)

# drawing the plots
fig.add_traces(
    [
        go.Histogram(x=x, hoverinfo="x", showlegend=False),
        go.Histogram(x=x_log, hoverinfo="x", showlegend=False),
        go.Histogram(x=x_square_root, hoverinfo="x", showlegend=False),
        go.Histogram(x=x_boxcox, hoverinfo="x", showlegend=False),
    ],
    rows=[1, 1, 2, 2],
    cols=[1, 2, 1, 2],
)

fig.update_layout(
    title=dict(
        text="GrLivArea with various Power Transforms",
        font=dict(family="Arial", size=20),
    ),
    showlegend=False,
    width=800,
    height=500,
)

fig.show()  # display figure
