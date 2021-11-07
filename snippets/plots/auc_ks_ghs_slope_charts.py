import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

class credit_model_eval:

    @staticmethod
    def roc_curve_graph(y, predict_proba, model_name='Modelo Sample'):
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

        f, ax = plt.subplots(figsize=(8,5))
        ax.plot('fpr', 'tpr', data=ROC, markersize = 8, color = 'navy', linewidth = 2)
        ax.plot(x, y, marker = '', markersize = 8, linestyle = '--', color = 'black', linewidth = 1)
        
        title = '{} - AUC: {:.4f}'.format(model_name, auc)
        ax.set_title(title, loc = 'center', fontsize = 12, fontweight = 0, color = 'black')
        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")
        return ax


    @staticmethod
    def ks_graph(gh_field, target_field, model_name = 'Modelo Sample'):
        """Generate KS graph with GH list of values and real target list of values

        Args:
            gh_field [(str)]: List of value of GH
            target_field [(str)]: List of real target value
            model_name (str, optional): Name of Model. Defaults to 'Modelo Sample'.

        Returns:
            object: Return graph of KS
        """        
        data = pd.crosstab(gh_field, target_field).reset_index()
        data['% 0'] = data[0] / data[0].sum()
        data['% 1'] = data[1] / data[1].sum()

        data['% 0 Acumulated'] = data['% 0'].cumsum()
        data['% 1 Acumulated'] = data['% 1'].cumsum()

        data['Distance'] = data['% 1 Acumulated'] - data['% 0 Acumulated']
    
        ks = data['Distance'].max()

        f, ax = plt.subplots(figsize=(8,5))
        ax.plot(data['GH'], '% 0 Acumulated', data=data, markersize = 8, color = 'navy', linewidth = 2)
        ax.plot(data['GH'], '% 1 Acumulated', data=data, markersize = 8, color = 'navy', linewidth = 2)

        title = '{} - KS: {:.4f}'.format(model_name, ks)
        ax.set_title(title, loc = 'center', fontsize = 12, fontweight = 0, color = 'black')
        ax.set_xlabel("GH")
        return ax


    @staticmethod
    def get_GHS(dataframe, n_ghs, score_field='score'):
        """Create GH list using dataframe e quantity of divisions

        Args:
            dataframe (pandas obj): Dataframe object containing proba predict
            n_ghs (int): Quantity of GHs to generate
            score_field (str, optional): Score field name. Defaults to 'score'.

        Returns:
            dataframe: Initial dataframe containing GH field as 'GH'
        """
        dataframe['GH'] = pd.qcut(dataframe[score_field],
                                q=[x / n_ghs for x in range(n_ghs)] + [1],
                                duplicates='drop')

        gh_dict = {x: str(idx + 1) for idx, x in enumerate(dataframe['GH'].unique().sort_values())}
        dataframe['GH'] = dataframe['GH'].map(gh_dict)
        return dataframe


    @staticmethod
    def GH_graph(gh_field, target_field, model_name = 'Modelo Sample'):
        """Create bar chart with target distribution by GH

        Args:
            gh_field [(str)]: List of value of GH
            target_field [(str)]: List of real target value
            model_name (str, optional): Name of model. Defaults to 'Modelo Sample'.

        Returns:
            object: Return Target distribution by GH chart
        """        
        data = pd.crosstab(gh_field, target_field).reset_index()

        data['% 0'] = round((data[0] / (data[0] + data[1])) * 100,1)
        data['% 1'] = round((data[1] / (data[0] + data[1])) * 100,1)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(data['GH'], data['% 0'], label='% 0')
        ax.bar(data['GH'], data['% 1'], bottom=data['% 0'], label='% 1')

        title = 'Distribuicao do {}'.format(model_name)
        ax.set_title(title, loc = 'center', fontsize = 12, fontweight = 0, color = 'black')

        ax.set_xlabel('GH')
        ax.set_ylabel('% Volume')
        ax.legend(loc='upper left')

        for idx, container in enumerate(ax.containers):
            if idx == 0:
                ax.bar_label(container, labels=data['% 0'], label_type='center')
            else:
                ax.bar_label(container, labels=data['% 1'], label_type='center')
        return ax


    @staticmethod
    def plot_slope_chart(names_1,
                        positions_1,
                        names_2,
                        positions_2,
                        modelo1 = 'Modelo 1',
                        modelo2 = 'Modelo 2',
                        figsize=(10,10)):
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

        df1 = pd.DataFrame({'ranking': positions_1,'descriptions': names_1})
        df1['column'] = 1

        df2 = pd.DataFrame({'ranking': positions_2,'descriptions': names_2})
        df2['column'] = 2

        dataset = pd.concat([df1, df2])
        
        descriptions = list(dataset['descriptions'].unique())

        fig, ax = plt.subplots(1, figsize)

        for desc in descriptions:
            df_plot = dataset[dataset['descriptions'] == desc]
            ax.plot(df_plot['column'], df_plot['ranking'], '-o', linewidth=7, markersize=10, alpha=0.5)

            if (df_plot['column'] == 1).any():
                name_plot = df_plot[df_plot['column'] == 1]
                ax.text(name_plot['column'].values[0] - 0.05, name_plot['ranking'].values[0], desc, ha='right')
            
            if (df_plot['column'] == 2).any():
                name_plot = df_plot[df_plot['column'] == 2]
                ax.text(name_plot['column'].values[0] + 0.05, name_plot['ranking'].values[0], desc, ha='left')

        ax.invert_yaxis()
        
        ax.set_xlim(0.5,2.5)
        ax.set_xticks([1,2])

        ax.set_xticklabels([modelo1, modelo2])

        ax.xaxis.grid(color='black', linestyle='solid', which='both', alpha=0.9)
        ax.yaxis.grid(color='black', linestyle='dashed', which='both', alpha=0.2)

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

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