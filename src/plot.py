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
import seaborn as sns                               # plotting      |
from scipy.interpolate import make_interp_spline    # trend line    |
from scipy.signal import savgol_filter              # trend line    |
# local imports ----------------------------------------------------
from helper import BOLD, PATH           # TUI, directories          |
# ------------------------------------------------------------------

def plot_distributions(dataframe: pd.DataFrame, metric: str, s_suff: str) -> None:

    global _S
    _S = s_suff

    dataframe = dataframe.copy()                    # prevent original dataframe being changed (it is mutable)
    dataframe.dropna(inplace=True)                  # drop test entries that do no have relevance scores
    dataframe = filter_rare_relevancies(dataframe)  # rare relevancies always need to be filtered
    
    for filter in [False, True]:
        if filter:
            if metric.startswith('sim'):
                continue                            # simple similarity columns should not be filtered
            dataframe = filter_low_similarities(dataframe, metric)
        
        avg_similarities: OrderedDict = calc_avg_similarities(dataframe, metric)
        create_area_plot(dataframe, metric, avg_similarities, filter)

def create_area_plot(dataframe: pd.DataFrame, metric: str, avg_similarities: dict, filter: bool) -> None:
    """Creates a seaborn `displot` and saves it to disk"""
    title: str = metric.replace('sim_sim_', 'Simple Similarity ').replace('sem_sim_', 'Semantic Similarity ')\
                       .replace('product_title', 'Product Title').replace('product_description', 'Product Description')
    alpha = 0.05 if (_S == '_sample') else 0.006
    
    rel, sim = list(avg_similarities.keys()), list(avg_similarities.values())
    X_Y_Spline = make_interp_spline(rel, savgol_filter(sim, 5, 3))
    X_ = np.linspace(min(rel), max(rel), 500); Y_ = X_Y_Spline(X_)
    
    area_plot = sns.displot(dataframe, x='relevance', y=metric,
                                kind='kde', fill=True, levels=15, cmap='viridis', thresh=0)
    area_plot.ax.scatter(dataframe['relevance'], dataframe[metric], color='white', alpha=alpha) # raw points
    area_plot.ax.plot(rel, sim, color='tab:red')                                                # raw averages
    area_plot.ax.plot(X_, Y_, color='tab:orange', linestyle=':')                                # smoothed averages
    area_plot.ax.set_ylabel('similarity score')
    f_suff = ' (filtered)' if filter else ''
    area_plot.ax.set_title(title+f_suff)
    f_suff = '_filtered' if filter else ''
    area_plot.fig.savefig(PATH('..',f'results{_S}',f'{metric}_plot{f_suff}.png'), bbox_inches='tight', dpi=300)

def calc_avg_similarities(dataframe: pd.DataFrame, metric: str) -> OrderedDict:
    """Calculates the average similarity scores of a given metric"""
    similarities, occurrences = {}, {}
    for _, row in dataframe.iterrows():
        rel, sim = row['relevance'], row[metric]
        try: similarities[rel] += (sim - similarities[rel])/occurrences[rel]; occurrences[rel] += 1
        except KeyError: similarities[rel] = sim; occurrences[rel] = 1
    return OrderedDict(sorted(similarities.items()))

def print_avg_similarities(dataframe: pd.DataFrame, col_name: str) -> None:
    """Prints raw data on the similarity scores of a metric that could also be plotted"""
    similarities = calc_avg_similarities(dataframe, col_name)
    print(BOLD(' rel |  sim  '))
    print(BOLD('-----+-------'))
    for rel in sorted(similarities.keys()):
        print(f'{rel:<4} {BOLD("|")} {round(similarities[rel], 3):<5}')

def filter_low_similarities(dataframe: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Filters out entries with unworkably low similarity scores"""
    dataframe = dataframe[getattr(dataframe, col_name) > 0.1]
    return dataframe

def filter_rare_relevancies(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filters out entries with relevancy scores that occur less than 5 times"""
    rel = dataframe.relevance
    dataframe = dataframe[(rel != 1.25) & (rel != 1.5) & (rel != 2.5)  & (rel != 2.75)]
    return dataframe
