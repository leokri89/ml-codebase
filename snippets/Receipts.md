
## models/train_lgboost.py
```python
import lightgbm as lgb


def train_lgb_model(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1", "rmse"},
        "learning_rate": 0.01,
        "num_leaves": 23,
        "num_iterations": 10000,
        "verbosity": -1,
    }
    m = lgb.train(
        params,
        train_set=lgb_train,
        valid_sets=lgb_eval,
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    return m
```
## models/train_prophet.py
```python
import fbprophet as pr

model = pr.Prophet()


def train_model(model, dataframe, periods=30, freq="1min"):
    model.fit(dataframe)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


forecast = train_model(model, dataframe, 30, "1min")
fig1 = model.plot(forecast)
```
## numerical_engineering/interaction.py
```python
# https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# creating dummy dataset
X = np.arange(10).reshape(5, 2)
X.shape
# >>> (5, 2)

# interactions between features only
interactions = PolynomialFeatures(interaction_only=True)
X_interactions = interactions.fit_transform(X)
X_interactions.shape
# >>> (5, 4)

# polynomial features
polynomial = PolynomialFeatures(5)
X_poly = polynomial.fit_transform(X)
X_poly.shape
# >>> (5, 6)
```
## numerical_engineering/power_transformation.py
```python
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
```
## numerical_engineering/quantization_adaptative_bin.py
```python
# https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import pandas as pd

# map the counts to quantiles (adaptive binning)
views_adaptive_bin = pd.qcut(views, 5, labels=False)

print(f"Adaptive bins: {views_adaptive_bin}")
# >>> Adaptive bins: [1 3 0 1 4 2 3 4 0 4 0 2 3 1]
```
## numerical_engineering/quantization_static_bin.py
```python
# https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import numpy as np

# 15 random integers from the "discrete uniform" distribution
ages = np.random.randint(0, 100, 15)

# evenly spaced bins
ages_binned = np.floor_divide(ages, 10)

print(f"Ages: {ages} \nAges Binned: {ages_binned} \n")
# >>> Ages: [97 56 43 73 89 68 67 15 18 36  4 97 72 20 35]
# Ages Binned: [9 5 4 7 8 6 6 1 1 3 0 9 7 2 3]

# numbers spanning several magnitudes
views = [300, 5936, 2, 350, 10000, 743, 2854, 9113, 25, 20000, 160, 683, 7245, 224]

# map count -> exponential width bins
views_exponential_bins = np.floor(np.log10(views))

print(f"Views: {views} \nViews Binned: {views_exponential_bins}")
# >>> Views: [300, 5936, 2, 350, 10000, 743, 2854, 9113, 25, 20000, 160, 683, 7245, 224]
# Views Binned: [2. 3. 0. 2. 4. 2. 3. 3. 1. 4. 2. 2. 3. 2.]
```
## numerical_engineering/scaling.py
```python
# https://www.evernote.com/l/Ap4HHraO7sxL16hpXgcVW0WWWbYO6c7nZW0/

import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

wine_json = load_wine()  # load in dataset

df = pd.DataFrame(
    data=wine_json["data"], columns=wine_json["feature_names"]
)  # create pandas dataframe

df["Target"] = wine_json["target"]  # created new column and added target labels

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
trace1 = go.Scatter(
    x=df_std[:, 0], y=df_std[:, 1], mode="markers", name="Standardized Scale"
)

trace2 = go.Scatter(
    x=df_minmax[:, 0], y=df_minmax[:, 1], mode="markers", name="MinMax Scale"
)

trace3 = go.Scatter(
    x=df_l2norm[:, 0], y=df_l2norm[:, 1], mode="markers", name="L2 Norm Scale"
)

trace4 = go.Scatter(
    x=df["alcohol"], y=df["malic_acid"], mode="markers", name="Original Scale"
)

layout = go.Layout(
    title="Effects of Feature scaling",
    xaxis=dict(title="Alcohol"),
    yaxis=dict(title="Malic Acid"),
)

data = [trace1, trace2, trace3, trace4]
fig = go.Figure(data=data, layout=layout)
fig.show()
```
## plots/auc_ks_ghs_slope_charts.py
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


class credit_model_eval:
    @staticmethod
    def roc_curve_graph(y, predict_proba, model_name="Modelo Sample"):
        """Generate Roc Auc Curve graph with score

        Args:
            y [(int)]: List of class
            predict_proba [(float)]: List of probability of the class with the greater label.
            model_name (str, optional): Name of model to be graph title. Defaults to 'Modelo Sample'.

        Returns:
            object: Return graph of Roc Auc Curve
        """

        fpr, tpr, thr = roc_curve(y, predict_proba)
        ROC = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        auc = roc_auc_score(y, predict_proba)

        x = np.linspace(0, 1, len(ROC))
        y = np.linspace(0, 1, len(ROC))

        f, ax = plt.subplots(figsize=(8, 5))
        ax.plot("fpr", "tpr", data=ROC, markersize=8, color="navy", linewidth=2)
        ax.plot(
            x, y, marker="", markersize=8, linestyle="--", color="black", linewidth=1
        )

        title = "{} - AUC: {:.4f}".format(model_name, auc)
        ax.set_title(title, loc="center", fontsize=12, fontweight=0, color="black")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")
        return ax

    @staticmethod
    def ks_graph(gh_field, target_field, model_name="Modelo Sample"):
        """Generate KS graph with GH list of values and real target list of values

        Args:
            gh_field [(str)]: List of value of GH
            target_field [(str)]: List of real target value
            model_name (str, optional): Name of Model. Defaults to 'Modelo Sample'.

        Returns:
            object: Return graph of KS
        """
        data = pd.crosstab(gh_field, target_field).reset_index()
        data["% 0"] = data[0] / data[0].sum()
        data["% 1"] = data[1] / data[1].sum()

        data["% 0 Acumulated"] = data["% 0"].cumsum()
        data["% 1 Acumulated"] = data["% 1"].cumsum()

        data["Distance"] = data["% 1 Acumulated"] - data["% 0 Acumulated"]

        ks = data["Distance"].max()

        f, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            data["GH"],
            "% 0 Acumulated",
            data=data,
            markersize=8,
            color="navy",
            linewidth=2,
        )
        ax.plot(
            data["GH"],
            "% 1 Acumulated",
            data=data,
            markersize=8,
            color="navy",
            linewidth=2,
        )

        title = "{} - KS: {:.4f}".format(model_name, ks)
        ax.set_title(title, loc="center", fontsize=12, fontweight=0, color="black")
        ax.set_xlabel("GH")
        return ax

    @staticmethod
    def get_GHS(dataframe, n_ghs, score_field="score"):
        """Create GH list using dataframe e quantity of divisions

        Args:
            dataframe (pandas obj): Dataframe object containing proba predict
            n_ghs (int): Quantity of GHs to generate
            score_field (str, optional): Score field name. Defaults to 'score'.

        Returns:
            dataframe: Initial dataframe containing GH field as 'GH'
        """
        dataframe["GH"] = pd.qcut(
            dataframe[score_field],
            q=[x / n_ghs for x in range(n_ghs)] + [1],
            duplicates="drop",
        )

        gh_dict = {
            x: str(idx + 1)
            for idx, x in enumerate(dataframe["GH"].unique().sort_values())
        }
        dataframe["GH"] = dataframe["GH"].map(gh_dict)
        return dataframe

    @staticmethod
    def GH_graph(gh_field, target_field, model_name="Modelo Sample"):
        """Create bar chart with target distribution by GH

        Args:
            gh_field [(str)]: List of value of GH
            target_field [(str)]: List of real target value
            model_name (str, optional): Name of model. Defaults to 'Modelo Sample'.

        Returns:
            object: Return Target distribution by GH chart
        """
        data = pd.crosstab(gh_field, target_field).reset_index()

        data["% 0"] = round((data[0] / (data[0] + data[1])) * 100, 1)
        data["% 1"] = round((data[1] / (data[0] + data[1])) * 100, 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(data["GH"], data["% 0"], label="% 0")
        ax.bar(data["GH"], data["% 1"], bottom=data["% 0"], label="% 1")

        title = "Distribuicao do {}".format(model_name)
        ax.set_title(title, loc="center", fontsize=12, fontweight=0, color="black")

        ax.set_xlabel("GH")
        ax.set_ylabel("% Volume")
        ax.legend(loc="upper left")

        for idx, container in enumerate(ax.containers):
            if idx == 0:
                ax.bar_label(container, labels=data["% 0"], label_type="center")
            else:
                ax.bar_label(container, labels=data["% 1"], label_type="center")
        return ax

    @staticmethod
    def plot_slope_chart(
        names_1,
        positions_1,
        names_2,
        positions_2,
        modelo1="Modelo 1",
        modelo2="Modelo 2",
        figsize=(10, 10),
    ):
        """Create slope chart

        Args:
            names_1 (list[str]): List of description names, must be the same size and sort of positions
            positions_1 (list[int]): List of rank numbers, must be the same size and sort of names
            names_2 (list[str]): List of description names, must be the same size and sort of positions
            positions_2 (list[int]): List of rank numbers, must be the same size and sort of names
            modelo1 (str, optional): Name of model in column 1. Defaults to 'Modelo 1'.
            modelo2 (str, optional): Name of model in column 2. Defaults to 'Modelo 2'.
            figsize (tuple, optional): Sige of plot. Defaults to (10,10).

        Returns:
            plot: Slope Chart
        """

        df1 = pd.DataFrame({"ranking": positions_1, "descriptions": names_1})
        df1["column"] = 1

        df2 = pd.DataFrame({"ranking": positions_2, "descriptions": names_2})
        df2["column"] = 2

        dataset = pd.concat([df1, df2])

        descriptions = list(dataset["descriptions"].unique())

        fig, ax = plt.subplots(1, figsize)

        for desc in descriptions:
            df_plot = dataset[dataset["descriptions"] == desc]
            ax.plot(
                df_plot["column"],
                df_plot["ranking"],
                "-o",
                linewidth=7,
                markersize=10,
                alpha=0.5,
            )

            if (df_plot["column"] == 1).any():
                name_plot = df_plot[df_plot["column"] == 1]
                ax.text(
                    name_plot["column"].values[0] - 0.05,
                    name_plot["ranking"].values[0],
                    desc,
                    ha="right",
                )

            if (df_plot["column"] == 2).any():
                name_plot = df_plot[df_plot["column"] == 2]
                ax.text(
                    name_plot["column"].values[0] + 0.05,
                    name_plot["ranking"].values[0],
                    desc,
                    ha="left",
                )

        ax.invert_yaxis()

        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])

        ax.set_xticklabels([modelo1, modelo2])

        ax.xaxis.grid(color="black", linestyle="solid", which="both", alpha=0.9)
        ax.yaxis.grid(color="black", linestyle="dashed", which="both", alpha=0.2)

        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)

        return ax

    @staticmethod
    def transform_proba_into_score(proba):
        """Convert probabilitie into Score

        Args:
            proba (int): Value of probability

        Returns:
            score: Value of probability converted into score
        """
        return int((1 - proba) * 1000)
```
## plots/distribution_charts.py
```python
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def distribution_graph(dataframe, plot_per_line):
    fig = plt.figure(figsize=(15, 60))
    sample = dataframe.columns.tolist()[:100]

    plot_column_num = plot_per_line
    plot_line_num = math.ceil(len(train.columns) / plot_column_num)

    for i in range(len(sample)):
        sns.set_style("dark")
        plt.subplot(plot_line_num, plot_column_num, i + 1)

        title = sample[i]
        plot = dataframe[title]

        plt.title(title, size=12, fontname="sans")
        a = sns.kdeplot(
            plot,
            color="#ff9900",
            shade=True,
            alpha=0.9,
            linewidth=1.5,
            edgecolor="black",
        )

        plt.ylabel("")
        plt.xlabel("")
        plt.xticks(fontname="sans")
        plt.yticks([])
        for j in ["right", "left", "top"]:
            a.spines[j].set_visible(False)
            a.spines["bottom"].set_linewidth(1.2)
    fig.tight_layout(h_pad=3)
    return plt
```
## plots/roc_auc_plot.py
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def roc_auc_graph(y_test, y_predict):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_predict)
    AUC = roc_auc_score(y_test, y_predict)

    data = dict(
        false_positive_rate=false_positive_rate, true_positive_rate=true_positive_rate
    )

    ROC = pd.DataFrame(data)
    x = np.linspace(0, 1, len(ROC))
    y = np.linspace(0, 1, len(ROC))

    plot_title = "AUC: {:.4f}".format(AUC)
    plt.title(plot_title, loc="center", fontsize=14, fontweight=0, color="black")

    plt.plot(
        "False Positive Rate",
        "True Positive Rate",
        data=ROC,
        marker="",
        markersize=8,
        color="navy",
        linewidth=3,
    )
    plt.plot(x, y, marker="", markersize=10, linestyle="--", color="black", linewidth=1)

    plt.ylabel("True positive")
    plt.xlabel("False Positive")
```
## snippets/distribution_charts.py
```python
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def distribution_graph(dataframe, plot_per_line):
    fig = plt.figure(figsize=(15, 60))
    sample = dataframe.columns.tolist()[:100]

    plot_column_num = plot_per_line
    plot_line_num = math.ceil(len(train.columns) / plot_column_num)

    for i in range(len(sample)):
        sns.set_style("dark")
        plt.subplot(plot_line_num, plot_column_num, i + 1)

        title = sample[i]
        plot = dataframe[title]

        plt.title(title, size=12, fontname="sans")
        a = sns.kdeplot(
            plot,
            color="#ff9900",
            shade=True,
            alpha=0.9,
            linewidth=1.5,
            edgecolor="black",
        )

        plt.ylabel("")
        plt.xlabel("")
        plt.xticks(fontname="sans")
        plt.yticks([])
        for j in ["right", "left", "top"]:
            a.spines[j].set_visible(False)
            a.spines["bottom"].set_linewidth(1.2)
    fig.tight_layout(h_pad=3)
    return plt
```
## snippets/fast_string_datetime_convert_pandas.py
```python
from datetime import datetime

import pandas as pd


def datetime_convert(year, month, day):
    try:
        return datetime(year=int(year), month=int(month), day=int(day))
    except:
        return datetime(year=1900, month=1, day=1)


df = pd.DataFrame({"col1": ["20210115", "20210116", "20210117", "20210118"]})
df["DATPRG"] = df["DATPRG"].apply(lambda x: datetime_convert(x[:4], x[4:6], x[6:8]))
```
## snippets/generate_pandas_profiling.py
```python
TRAIN_SET = "./datasets/train.csv"
TEST_SET = "./datasets/test.csv"

PROFILING_CONFIG = "./profiling_config.yml"

import pandas as pd
import pandas_profiling as pdp

df = pd.read_csv(TRAIN_SET)

profile = pdp.ProfileReport(df, config_file=PROFILING_CONFIG)
profile.to_file("profiling.html")
```
## snippets/get_profiling_categorical_vars.py
```python
def profiling_categoricals(df):
    """Profiling a numerical dataframe

    Args:
        df (dataframe):

    Raises:
        RuntimeError: Error

    Returns:
        dataframe: A dataframe with categorical profiling
    """
    types = df.dtypes
    missing = round((df.isnull().sum() / df.shape[0]), 3) * 100
    uniques = df.apply(lambda x: x.unique())
    return pd.DataFrame({"Types:": types, "Missings%": missing, "Uniques": uniques})
```
## snippets/get_profiling_numerical_vars.py
```python
import pandas as pd


def outlier_calc(x):
    return (x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)))) | (
        x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))
    )


def profiling_numericals(df):
    """Profiling a numerical dataframe

    Args:
        df (dataframe):

    Raises:
        RuntimeError: Error

    Returns:
        dataframe: A dataframe with numerical profiling
    """
    types = df.dtypes
    missing = round((df.isnull().sum() / df.shape[0]), 3) * 100
    min = df.apply(lambda x: round(x.min()))
    max = df.apply(lambda x: round(x.max()))
    mean = df.apply(lambda x: round(x.mean()))
    outliers = df.apply(lambda x: sum(outlier_calc(x)))
    return pd.DataFrame(
        {
            "Types:": types,
            "Missings%": missing,
            "Min#": min,
            "Max#": max,
            "mean": mean,
            "Outliers#": outliers,
        }
    ).transpose()
```
## snippets/get_relevant_features_numin_numout.py
```python
from sklearn.feature_selection import (SelectKBest, SelectPercentile,
                                       f_regression)


def get_relevant_numin_numtarget(X, y, percentage_features=90, absolute_features=None):
    if absolute_features:
        if absolute_features > len(X.columns):
            absolute_features = len(X.columns)
        fs = SelectKBest(score_func=f_regression, k=absolute_features)
    else:
        fs = SelectPercentile(score_func=f_regression, percentile=percentage_features)

    fs.fit(X, y)
    data = [
        [X.columns[n], v, fs.scores_[n], fs.pvalues_[n]]
        for n, v in enumerate(fs.get_support())
    ]
    result = pd.DataFrame(data, columns=["column", "selected", "score", "pvalue"])
    return result


teste = geral[~geral["SalePrice"].isna()]

# get 20 features of relevant features in data
get_relevant_numin_numtarget(
    teste[var_numerica].fillna(0), teste["SalePrice"], absolute_features=20
)

# Get 90% of relevant features in data
get_relevant_numin_numtarget(
    teste[var_numerica].fillna(0), teste["SalePrice"], percentage_features=90
)
```
## snippets/get_relevant_numin_cattarget.py
```python
from sklearn.datasets import make_classification
from sklearn.feature_selection import (SelectKBest, SelectPercentile,
                                       f_classif, f_regression)

X, y = make_classification(n_samples=100, n_features=20, n_informative=2)


def get_relevant_numin_cattarget(X, y, percentage_features=90, absolute_features=None):
    if absolute_features:
        if absolute_features > len(X.columns):
            absolute_features = len(X.columns)
        fs = SelectKBest(score_func=f_classif, k=absolute_features)
    else:
        fs = SelectPercentile(score_func=f_classif, percentile=percentage_features)

    fs.fit(X, y)
    data = [
        [X.columns[n], v, fs.scores_[n], fs.pvalues_[n]]
        for n, v in enumerate(fs.get_support())
    ]
    result = pd.DataFrame(data, columns=["column", "selected", "score", "pvalue"])
    return result


get_relevant_numin_cattarget(pd.DataFrame(X), y, 10)
```
## snippets/get_short_url_tinyurl.py
```python
import requests


def get_shorturl(token, url):
    response = requests.post(
        "https://api.tinyurl.com/create",
        headers={"Authorization": token},
        data={"url": url},
    )
    return response.json()


token = ""
```
## snippets/logging_format.py
```python
import logging

logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)

logging.info("Downloading dados.")
logging.warning("warning")
logging.error("error")
```
## snippets/markdown_format.py
```python
from glob import glob

import isort
from black import FileMode, format_str


def main():
    CODE_PATH = r"E:\repositorio\github\ml-codebase\snippets"
    markdown_name = "Receipts.md"
    files = [
        file
        for file in glob(CODE_PATH + "\*", recursive=True)
        + glob(CODE_PATH + "\*\*", recursive=True)
        if ".py" in file
    ]
    code_list = [segment_file(file) for file in files]
    code_list = sorted(code_list, key=lambda k: k["segment"] + ":" + k["file"])
    for key, code in enumerate(code_list):
        if key == 0:
            save_to_markdown(code, markdown_name, "w+")
        else:
            save_to_markdown(code, markdown_name, "a")


def segment_file(file):
    file_list = file.split("\\")
    segment = file_list[len(file_list) - 2]
    file_name = file_list[len(file_list) - 1]

    with open(file, "r") as fopened:
        code = fopened.read()

    return {"segment": segment, "file": file_name, "code": code}


def save_to_markdown(data, markdown_name="teste.md", mode="w+"):
    segment = data["segment"]
    file = data["file"]
    code = data["code"]

    with open(markdown_name, mode) as fopen:
        try:
            code_formated = format_str(code, mode=FileMode())
        except:
            code_formated = code

        try:
            code_formated = isort.code(code_formated)
        except:
            code_formated = code_formated

        form = f"\n## {segment}/{file}\n```python\n{code_formated}```"
        fopen.write(form)


if __name__ == "__main__":
    main()
```
## snippets/mlflow_autolog.py
```python
import logging
import time
import warnings

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"
SAMPLE_PATH = "./dataset/sample_submission.csv"
SEED = 10


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri("http://127.0.0.1:5000")


def train_eval_model(
    model, train_features_set, train_target_set, test_features_set, test_target_set
):
    """[summary]

    Args:
        model ([type]): [description]
        train_features_set ([type]): [description]
        train_target_set ([type]): [description]
        test_features_set ([type]): [description]
        test_target_set ([type]): [description]

    Returns:
        [type]: [description]
    """
    model.fit(train_features_set, train_target_set)
    # predicts = model.predict(test_features_set)
    proba_predicts = model.predict_proba(test_features_set)[:, 1]
    auc = roc_auc_score(test_target_set, proba_predicts)
    return model, auc


train = pd.read_csv(TRAIN_PATH, low_memory=False)
target = train["target"].copy()
train.drop(["id", "target"], inplace=True, axis=1)

continuous = list(train.columns[train.dtypes == "float64"])
discrets = list(train.columns[(train.dtypes == "int64")])

skf = StratifiedKFold(n_splits=3, random_state=SEED, shuffle=True)
splits = [[x, y] for x, y in skf.split(train, target)]

mlflow.sklearn.autolog(silent=True, disable=True)

kbd = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="kmeans")
kbd.fit(train[continuous], target)
modified_set = pd.DataFrame(kbd.transform(train[continuous]), columns=continuous)
modified_set[discrets] = train[discrets]

feature_set = modified_set
target_set = target

with mlflow.start_run(nested=True) as run:
    mlflow.sklearn.autolog(silent=True, disable=False)

    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)

    for idx, split in enumerate(splits):
        start_time = time.time()
        idx_train, idx_test = split[0], split[1]

        with mlflow.start_run(run_name=f"split_{str(idx)}", nested=True) as split_run:
            trained_model, auc_result = train_eval_model(
                rfc,
                feature_set.loc[idx_train],
                target_set.loc[idx_train],
                feature_set.loc[idx_test],
                target_set.loc[idx_test],
            )
            mlflow.log_metric("test_roc_auc_score", auc_result)
            mlflow.log_metric("execution_time", time.time() - start_time)

    mlflow.sklearn.autolog(silent=True, disable=True)
```
## snippets/remove_duplicated_low_memory.py
```python
seen = set()

flist = ["/path/to/file/dataset_{}.json".format(x) for x in range(8)]

with open("/path/to/file/result_0.json", "w") as fout:
    for file in flist:
        with open(file) as fin:
            for line in fin:
                id = hash(line)
                if id not in seen:
                    fout.write(line)
                    seen.add(id)
```
## snippets/remove_high_null_columns.py
```python
def clean_null_columns(df, threshold):
    """Profiling a numerical dataframe

    Args:
        df (dataframe):

    Raises:
        RuntimeError: Error

    Returns:
        dataframe: A dataframe without columns above threshold of null
    """
    col_list = (~(df.isnull().sum() / df.shape[0] * 100 > threshold)).tolist()
    return df.iloc[:, col_list].copy()


workset = clean_null_columns(df, 80)
```
## snippets/remove_highcardinality.py
```python
import pandas as pd


def remove_highcardinality(dataframe):
    nunique_res = dataframe.nunique()
    to_remove = [
        n
        for n, v in enumerate(nunique_res)
        if (float(v) / dataframe.shape[0]) * 100 <= 0.1
    ]
    dataframe.drop(columns=dataframe.columns[to_remove], axis=1, inplace=True)
    return dataframe


geral = remove_highcardinality(geral)
```
## snippets/remove_univars.py
```python
import pandas as pd


def remove_univars(dataframe):
    nunique_res = dataframe.nunique()
    to_remove = [n for n, v in enumerate(nunique_res) if v == 1]
    dataframe.drop(to_remove, axis=1, inplace=True)
    return dataframe


geral = remove_univars(geral)
```
## snippets/requests_with_authentication.py
```python
import datetime
import json

import requests

login_url = "https://engine.bompracredito.com.br/api/CredMarketApi/Login"
authenticated_url = "https://ckqry.bompracredito.com.br/milestoneStats"

payload = json.dumps({"user": "*", "password": "*"})
headers = {"Content-Type": "application/json"}

with requests.Session() as sesh:
    response = sesh.post(login_url, data=payload, headers=headers)
    session_id = json.loads(response.text).get("sessionId")

    payload = json.dumps(
        {"match": {"DayMonthYear": {"$gte": "2021-06-01"}}, "InvestorId": 1}
    )
    headers = {"session-id": session_id, "Content-Type": "application/json"}

    response = requests.request(
        "POST", authenticated_url, headers=headers, data=payload, verify=False
    )

    data = json.loads(response.text)
```
## snippets/roc_auc_plot.py
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def roc_auc_graph(y_test, y_predict):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_predict)
    AUC = roc_auc_score(y_test, y_predict)

    data = dict(
        false_positive_rate=false_positive_rate, true_positive_rate=true_positive_rate
    )

    ROC = pd.DataFrame(data)
    x = np.linspace(0, 1, len(ROC))
    y = np.linspace(0, 1, len(ROC))

    plot_title = "AUC: {:.4f}".format(AUC)
    plt.title(plot_title, loc="center", fontsize=14, fontweight=0, color="black")

    plt.plot(
        "False Positive Rate",
        "True Positive Rate",
        data=ROC,
        marker="",
        markersize=8,
        color="navy",
        linewidth=3,
    )
    plt.plot(x, y, marker="", markersize=10, linestyle="--", color="black", linewidth=1)

    plt.ylabel("True positive")
    plt.xlabel("False Positive")
```
## snippets/timer_decorator.py
```python
from time import sleep, time


def timer(func):
    def func_wrap(*args, **kwargs):
        tstart = time()
        result = func(*args, **kwargs)
        tend = time()
        print(f"Function {func.__name__!r} executed in {(tend-tstart):.3f}s")
        return result

    return func_wrap


@timer
def looping(n):
    for i in range(n):
        sleep(1)


looping(10)
```
## snippets/train_lgboost.py
```python
import lightgbm as lgb


def train_lgb_model(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1", "rmse"},
        "learning_rate": 0.01,
        "num_leaves": 23,
        "num_iterations": 10000,
        "verbosity": -1,
    }
    m = lgb.train(
        params,
        train_set=lgb_train,
        valid_sets=lgb_eval,
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    return m
```
## snippets/train_prophet.py
```python
import fbprophet as pr

model = pr.Prophet()


def train_model(model, dataframe, periods=30, freq="1min"):
    model.fit(dataframe)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


forecast = train_model(model, dataframe, 30, "1min")
fig1 = model.plot(forecast)
```
## snippets/value_mapper.py
```python
def value_mapper(df, fieldlist, value_dict):
    for col in fieldlist:
        df[col] = df[col].map(value_dict).fillna(0)
    return df


workset = value_mapper(
    workset,
    [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
    ],
    {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
)

workset = value_mapper(
    workset,
    ["BsmtFinType1", "BsmtFinType2"],
    {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "UnF": 1, "NA": 0},
)

workset = value_mapper(
    workset, ["GarageFinish"], {"Fin": 3, "RFn": 2, "UnF": 1, "NA": 0}
)

workset = value_mapper(workset, ["CentralAir"], {"Y": 1, "N": 0})
```