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
