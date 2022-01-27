## Distribution Chart
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