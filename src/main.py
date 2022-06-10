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
                        calc_simple_similarity)                 # ...               |
from plot import plot_distributions                             # plotting          |
# ----------------------------------------------------------------------------------

def main():
    
    # ------------------ #
    # STANDARD PROCEDURE #
    # ------------------ #

    datasets = ['train', 'product_descriptions']
    nlp: spacy.Language = spacy.load('en_core_web_lg')

    print(f'pandas: v{pd.__version__}, spaCy: v{spacy.__version__}')

    arg_parser = argparse.ArgumentParser()
    s_suff, p_flag, c_flag, d_flag = argparse_wrapper(arg_parser)
    suppress_W008()
    fix_dirs(s_suff, p_flag)
    print_pipeline(datasets, p_flag, c_flag, d_flag)

    timer = Timer(first_process='reading original csv files')
    dataframes: list[pd.DataFrame] = load_dataframes(datasets, s_suff)

    dataframe = dataframes.pop(0)   # "train" is the fundamental dataframe

    for df in dataframes:
        dataframe = pd.merge(dataframe, df, how='left', on='product_uid')
    
    parsable_cols = [col for col in dataframe.columns if (dataframe[col].dtype == object)]


    if p_flag:
        create('docbins')
    # --------------------------------- #
    # PARSING STRING DATA TO SPACY DOCS #
    # --------------------------------- #
        
        for col in parsable_cols:
            timer(f'parsing {col}')
            s: pd.Series = dataframe[col]
            db: spacy.tokens.DocBin = parse_data(s, nlp)
            timer('saving parsed data to disk')
            store_as_docbin(db, s.name)
            del s, db   # help Python with garbage collection
    
    parsable_cols.remove('search_term')

    if c_flag:
        require('docbins')
        create('arrays')
    # ----------------------------- #
    # CALCULATING SIMILARITY SCORES #
    # ----------------------------- #

        timer('reading search_term spacy docs')
        docs: list[spacy.tokens.Doc] = load_docs('search_term', nlp)
        dataframe['search_term'] = docs
        
        for col in parsable_cols:
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


    if d_flag:
        require('arrays')
    # ---------------------- #
    # PLOTTING DISTRIBUTIONS #
    # ---------------------- #

        translate = lambda x: x.replace('sim_sim_', 'simple similarity ').replace('sem_sim_', 'semantic similarity ')\
                            .replace('product_title', 'product title').replace('product_description', 'product description')

        for col in parsable_cols:
            for sim_kind in ['sem', 'sim']:
                metric: str = f'{sim_kind}_sim_{col}'
                timer(f'creating {translate(metric)} plots')
                array = load_array(f'{metric}')
                dataframe[f'{metric}'] = array[:,1]
                plot_distributions(dataframe, metric, s_suff)

    timer()


if __name__ == "__main__":
    main()
