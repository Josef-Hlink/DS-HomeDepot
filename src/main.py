"""
Main
===
Script that should be run to generate all results.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------
import argparse         # specifying args from command line |
# dependencies ---------------------------------------------
import pandas as pd     # dataframes                        |
import spacy            # natural language processing       |
# local imports --------------------------------------------------------------------
from helper import (argparse_wrapper, suppress_W008,            # general utilities |
                    fix_dirs, print_pipeline, Timer)            # ...               |
from datamanager import (load_dataframes, create, require,      # data management   |
                         store_as_docbin, load_docs,            # ...               |
                         store_as_array, load_array)            # ...               |
from processing import (parse_data, calc_semantic_similarity,   # processing data   |
                        calc_simple_similarity, calc_length)    # ...               |
from plot import plot_distributions                             # plotting          |
from model import train_and_test                                # regression model  |
# ----------------------------------------------------------------------------------

def main():
    
    # ------------------ #
    # STANDARD PROCEDURE #
    # ------------------ #

    datasets = ['train', 'product_descriptions']
    nlp: spacy.Language = spacy.load('en_core_web_lg')

    print(f'pandas: v{pd.__version__}, spaCy: v{spacy.__version__}')

    arg_parser = argparse.ArgumentParser()
    s_suff, p_flag, c_flag, d_flag, t_flag = argparse_wrapper(arg_parser)
    suppress_W008()
    fix_dirs(s_suff)
    print_pipeline(datasets, p_flag, c_flag, d_flag, t_flag)

    timer = Timer(first_process='reading original csv files')
    dataframes: list[pd.DataFrame] = load_dataframes(datasets, s_suff)
    df_train, df_prod_desc = dataframes
    dataframe = pd.merge(df_train, df_prod_desc, how='left', on='product_uid')


    if p_flag:
        create('docbins')
        # --------------------------------- #
        # PARSING STRING DATA TO SPACY DOCS #
        # --------------------------------- #
        
        for col in ['search_term', 'product_title', 'product_description']:
            timer(f'parsing {col}')
            s: pd.Series = dataframe[col]
            db: spacy.tokens.DocBin = parse_data(s, nlp)
            timer('saving parsed data to disk')
            store_as_docbin(db, s.name)
            dataframe.drop(col, axis=1, inplace=True)
            del s, db   # help Python with garbage collection


    if c_flag:
        require('docbins')
        create('arrays')
        # ----------------------------- #
        # CALCULATING SIMILARITY SCORES #
        # ----------------------------- #

        timer('reading search_term spacy docs')
        dataframe['search_term'] = load_docs('search_term', nlp)
        
        timer('calculating query length')
        dataframe['len_of_query'] = calc_length(dataframe['search_term'])
        store_as_array((dataframe['relevance'], dataframe['len_of_query']))
        dataframe.drop('len_of_query', axis=1, inplace=True)

        for col in ['product_title', 'product_description']:
            timer(f'reading {col} spacy docs')
            docs: list[spacy.tokens.Doc] = load_docs(col, nlp)
            dataframe[col] = docs

            timer(f'calculating semantic similarity search_term <-> {col}')
            dataframe[f'zipped_{col}'] = tuple(zip(dataframe['search_term'], dataframe[col]))
            dataframe[f'sem_sim_{col}'] = calc_semantic_similarity(dataframe[f'zipped_{col}'])

            timer(f'calculating simple similarity search_term <-> {col}')
            dataframe[f'sim_sim_{col}'] = calc_simple_similarity(dataframe[f'zipped_{col}'])

            store_as_array((dataframe['relevance'], dataframe[f'sem_sim_{col}']))
            store_as_array((dataframe['relevance'], dataframe[f'sim_sim_{col}']))
            dataframe.drop([col, f'zipped_{col}', f'sem_sim_{col}', f'sim_sim_{col}'], axis=1, inplace=True)
        
        dataframe.drop('search_term', axis=1, inplace=True)


    if d_flag:
        require('arrays')
        # ---------------------- #
        # PLOTTING DISTRIBUTIONS #
        # ---------------------- #

        translate = lambda x: x.replace('sim_sim_', 'simple similarity ')\
                               .replace('sem_sim_', 'semantic similarity ')

        for col in ['product_title', 'product_description']:
            for sim_kind in ['sem', 'sim']:
                metric: str = f'{sim_kind}_sim_{col}'
                timer(f'creating {translate(metric)} plots')
                array = load_array(f'{metric}')
                dataframe[f'{metric}'] = array[:,1]
                plot_distributions(dataframe, metric, s_suff)
                dataframe.drop(metric, axis=1, inplace=True)

    RMSE = None
    if t_flag:
        require('arrays')
        # -------------------------- #
        # TRAINING AND TESTING MODEL #
        # -------------------------- #

        timer('loading in all numerical data')
        dataframe['len_of_query'] = load_array('len_of_query')[:,1]

        for col in ['product_title', 'product_description']:
            for sim_kind in ['sem', 'sim']:
                metric: str = f'{sim_kind}_sim_{col}'
                array = load_array(f'{metric}')
                dataframe[f'{metric}'] = array[:,1]

        timer('training and testing')
        RMSE = train_and_test(dataframe)


    timer()
    if RMSE is not None: print(f'RMSE: {RMSE:.5f}')
    

if __name__ == "__main__":
    main()
