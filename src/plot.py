"""
Plot
===
Functions pertaining to plotting results.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ------------------------------
from collections import OrderedDict     # trend line    |
# dependencies -----------------------------------------------------
import numpy as np                                  # arrays        |
import pandas as pd                                 # dataframes    |
import matplotlib.pyplot as plt                     # plotting      |
import seaborn as sns                               # plotting      |
from scipy.interpolate import make_interp_spline    # trend line    |
from scipy.signal import savgol_filter              # trend line    |
# local imports ----------------------------------------------------------------------------
from helper import BOLD, PATH                                           # TUI, directories  |
from processing import filter_rare_relevancies, filter_low_similarities # filtering data    |
# ------------------------------------------------------------------------------------------

def plot_distributions(dataframe: pd.DataFrame, metric: str, s_suff: str) -> None:
    """
    Creates all distribution plots for a given metric (wrapper for `create_area_plot()`).
    ### params
        - dataframe: the full pandas `DataFrame` with all data inside
        - metric: the name of the column of which the similarity scores need to be plotted
        - s_suff: lets plot.py know on what dataset the experiment is running
    """
    global _S
    _S = s_suff

    dataframe = dataframe.copy()                    # prevent original dataframe being changed (it is mutable)
    dataframe = filter_rare_relevancies(dataframe)  # rare relevancies always need to be filtered
    
    for filter in [False, True]:
        if filter:
            if metric.startswith('sim'):
                continue                            # simple similarity columns should not be filtered
            dataframe = filter_low_similarities(dataframe, metric)
        
        avg_similarities: OrderedDict = calc_avg_similarities(dataframe, metric)
        create_area_plot(dataframe, metric, avg_similarities, filter)

def create_area_plot(dataframe: pd.DataFrame, metric: str, avg_similarities: dict, filter: bool) -> None:
    """
    Creates a seaborn `displot` and saves it to disk.
        - dataframe: the full (filtered) pandas `DataFrame` with all data inside
        - metric: the name of the column of which the similarity scores need to be plotted
        - avg_similarities: the average similarities of that column
        - filter: flag that indicates whether the data has been filtered or not
    """
    title: str = metric.replace('sim_sim_', 'Simple Similarity ').replace('sem_sim_', 'Semantic Similarity ')\
                       .replace('product_title', 'Product Title').replace('product_description', 'Product Description')
    alpha = 0.05 if (_S == '_sample') else 0.006
    
    rel, sim = list(avg_similarities.keys()), list(avg_similarities.values())
    X_Y_Spline = make_interp_spline(rel, savgol_filter(sim, 5, 3))
    X_ = np.linspace(min(rel), max(rel), 500); Y_ = X_Y_Spline(X_)
    
    area_plot = sns.displot(dataframe, x='relevance', y=metric,
                                kind='kde', fill=True, levels=15, cmap='viridis', thresh=0)
    area_plot.ax.scatter(dataframe['relevance'], dataframe[metric], color='white', alpha=alpha, label='raw data points')
    area_plot.ax.plot(rel, sim, color='tab:red', label='raw average')
    area_plot.ax.plot(X_, Y_, color='tab:orange', linestyle=':', label='smoothed average')
    area_plot.ax.set_ylabel('similarity score')
    area_plot.ax.legend(scatterpoints=20, labelcolor='white', facecolor='black', framealpha=0.5)
    f_suff = ' (filtered)' if filter else ''
    area_plot.ax.set_title(title+f_suff)
    f_suff = '_filtered' if filter else ''
    area_plot.fig.savefig(PATH('..',f'results{_S}',f'{metric}_plot{f_suff}.png'), bbox_inches='tight', dpi=300)

def calc_avg_similarities(dataframe: pd.DataFrame, metric: str) -> OrderedDict:
    """
    Calculates the average similarity scores of a given metric.
    ### params
        - dataframe: the full pandas `DataFrame` with all data inside
        - metric: the name of the column of which the similarity scores need to be averaged
    
    ### returns
        - an ordered dictionary with:
            * keys: relevancy scores (in increasing order)
            * the corresponding averaged similarity scores
    """
    similarities, occurrences = {}, {}
    for _, row in dataframe.iterrows():
        rel, sim = row['relevance'], row[metric]
        try: similarities[rel] += (sim - similarities[rel])/occurrences[rel]; occurrences[rel] += 1
        except KeyError: similarities[rel] = sim; occurrences[rel] = 1
    return OrderedDict(sorted(similarities.items()))

def print_avg_similarities(dataframe: pd.DataFrame, col_name: str) -> None:
    """Prints data on the average similarity scores of a metric that could also be plotted"""
    similarities = calc_avg_similarities(dataframe, col_name)
    print(BOLD(' rel |  sim  '))
    print(BOLD('-----+-------'))
    for rel in sorted(similarities.keys()):
        print(f'{rel:<4} {BOLD("|")} {round(similarities[rel], 3):<5}')

def plot_feature_importances(features: list[str], importances: list[float]) -> None:
    """Plots the importance of each feature that was used in the regression model"""

    translate = lambda feature: feature.replace('sim_sim_', 'simple sim. ').replace('sem_sim_', 'semantic sim. ')\
                .replace('product_title', 'prod. title').replace('product_description', 'prod. descr.')\
                .replace('len_of_query', 'query length')
    
    x_labels = list(map(translate, features))
    fig, ax = plt.subplots()
    ax.bar(x_labels, importances)
    ax.set_ylabel('importance')
    ax.set_title('Feature Importances')
    for label in ax.get_xticklabels():
        label.set_rotation(-45)
        label.set_ha('left')
    
    fig.savefig(PATH('..',f'results{_S}','feature_importances.png'), bbox_inches='tight', dpi=300)
