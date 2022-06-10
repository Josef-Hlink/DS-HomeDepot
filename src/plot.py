"""
Plot
===
Functions pertaining to plotting results.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ------------------------------
import os                               # directories   |
from collections import OrderedDict     # trend line    |
# dependencies -----------------------------------------------------
import numpy as np                                  # arrays        |
import pandas as pd                                 # dataframes    |
import seaborn as sns                               # plotting      |
from scipy.interpolate import make_interp_spline    # trend line    |
# ------------------------------------------------------------------

def plot_distribution(dataframe: pd.DataFrame, col_name: str, s_suff: str) -> None:

    dataframe = filter_rare_relevancies(dataframe)
    
    avg_similarities: OrderedDict = calc_avg_similarities(dataframe, col_name)
    rel, sim = list(avg_similarities.keys()), list(avg_similarities.values())
    X_Y_Spline = make_interp_spline(rel, sim)
    X_ = np.linspace(min(rel), max(rel), 500); Y_ = X_Y_Spline(X_)

    area_plot = sns.displot(dataframe, x='relevance', y='sim_'+col_name, kind='kde', fill=True)
    area_plot.ax.plot(X_, Y_, color='tab:orange')
    area_plot.fig.savefig(os.path.join(os.getcwd(),'..','results',f'{col_name}_area_plot1{s_suff}.png'), dpi=300)

    dataframe = filter_low_similarities(dataframe)
    
    avg_similarities: OrderedDict = calc_avg_similarities(dataframe, col_name)
    rel, sim = list(avg_similarities.keys()), list(avg_similarities.values())
    X_Y_Spline = make_interp_spline(rel, sim)
    X_ = np.linspace(min(rel), max(rel), 500); Y_ = X_Y_Spline(X_)

    area_plot = sns.displot(dataframe, x='relevance', y='sim_'+col_name, kind='kde', fill=True)
    area_plot.ax.plot(X_, Y_, color='tab:orange')
    area_plot.fig.savefig(os.path.join(os.getcwd(),'..','results',f'{col_name}_area_plot2{s_suff}.png'), dpi=300)

def calc_avg_similarities(dataframe: pd.DataFrame, col_name: str) -> OrderedDict:
    """Calculates the average similarity scores of a given metric"""
    similarities, occurrences = {}, {}
    for _, row in dataframe.iterrows():
        rel, sim = row['relevance'], row['sim_'+col_name]
        try: similarities[rel] += (sim - similarities[rel])/occurrences[rel]; occurrences[rel] += 1
        except KeyError: similarities[rel] = sim; occurrences[rel] = 1
    return OrderedDict(sorted(similarities.items()))

def print_avg_similarities(dataframe: pd.DataFrame, col_name: str) -> None:
    """Prints raw data on the similarity scores of a metric that could also be plotted"""
    similarities = calc_avg_similarities(dataframe, col_name)
    print(' rel |  sim  ')
    print('-----+-------')
    for rel in sorted(similarities.keys()):
        print(f'{rel:<4} | {round(similarities[rel], 3):<5}')

def filter_low_similarities(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filters out entries with unworkably low similarity scores"""
    # NOTE not modular
    dataframe = dataframe[dataframe.sim_product_title > 0.1]
    dataframe = dataframe[dataframe.sim_product_description > 0.1]
    return dataframe

def filter_rare_relevancies(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filters out entries with relevancy scores that occur less than 5 times"""
    rel = dataframe.relevance
    dataframe = dataframe[(rel != 1.25) & (rel != 1.5) & (rel != 2.5)  & (rel != 2.75)]
    return dataframe
