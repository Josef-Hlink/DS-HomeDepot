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
# local imports ----------------------------------------------------
from helper import BOLD, PATH           # TUI, directories          |
# ------------------------------------------------------------------

def plot_distribution(dataframe: pd.DataFrame, metric: str, s_suff: str) -> None:

    _S = s_suff

    dataframe = filter_rare_relevancies(dataframe)
    
    title: str = metric.replace('sim_sim_', 'Simple Similarity ').replace('sem_sim_', 'Semantic Similarity ')
    title: str = title.replace('product_title', 'Product Title').replace('product_description', 'Product Description')
    
    avg_similarities: OrderedDict = calc_avg_similarities(dataframe, metric)
    rel, sim = list(avg_similarities.keys()), list(avg_similarities.values())
    X_Y_Spline = make_interp_spline(rel, sim)
    X_ = np.linspace(min(rel), max(rel), 500); Y_ = X_Y_Spline(X_)

    area_plot = sns.displot(dataframe, x='relevance', y=metric,
                            kind='kde', fill=True, levels=15, cmap='viridis', thresh=0)
    area_plot.ax.scatter(dataframe['relevance'], dataframe[metric], color='white', alpha=0.05)
    area_plot.ax.plot(rel, sim, color='tab:orange', linestyle=':')
    area_plot.ax.plot(X_, Y_, color='tab:orange')
    area_plot.ax.set_ylabel('similarity score')
    area_plot.ax.set_title(title)
    area_plot.fig.savefig(PATH('..',f'results{_S}',f'{metric}_plot.png'), bbox_inches='tight', dpi=300)

    if metric.startswith('sim'):
        return  # simple similarity columns do not need to be filtered
    
    dataframe = filter_low_similarities(dataframe, metric)
    
    avg_similarities: OrderedDict = calc_avg_similarities(dataframe, metric)
    rel, sim = list(avg_similarities.keys()), list(avg_similarities.values())
    X_Y_Spline = make_interp_spline(rel, sim)
    X_ = np.linspace(min(rel), max(rel), 500); Y_ = X_Y_Spline(X_)

    area_plot = sns.displot(dataframe, x='relevance', y=metric,
                            kind='kde', fill=True, levels=15, cmap='viridis', thresh=0)
    area_plot.ax.scatter(dataframe['relevance'], dataframe[metric], color='white', alpha=0.05)
    area_plot.ax.plot(rel, sim, color='tab:orange', linestyle=':')
    area_plot.ax.plot(X_, Y_, color='tab:orange')
    area_plot.ax.set_ylabel('similarity score')
    area_plot.ax.set_title(f'{title} (filtered)')
    area_plot.fig.savefig(PATH('..',f'results{_S}',f'{metric}_plot_filtered.png'), bbox_inches='tight')

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
    # NOTE not modular
    dataframe = dataframe[getattr(dataframe, col_name) > 0.1]
    # dataframe = dataframe[dataframe.sem_sim_product_description > 0.1]
    return dataframe

def filter_rare_relevancies(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filters out entries with relevancy scores that occur less than 5 times"""
    rel = dataframe.relevance
    dataframe = dataframe[(rel != 1.25) & (rel != 1.5) & (rel != 2.5)  & (rel != 2.75)]
    return dataframe
