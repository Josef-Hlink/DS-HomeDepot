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
# local imports --------------------------------------------------------------
from helper import (argparse_wrapper, suppress_W008,     # general utilities  |
                    fix_dirs, print_pipeline, Timer)     # ""                 |
from datamanager import (load_dataframes, parse_data,    # data management    |
                         store_as_docbin, load_docs)     # ""                 |
# ----------------------------------------------------------------------------

def main():
    
    datasets = ['train', 'product_descriptions']
    nlp: spacy.Language = spacy.load('en_core_web_lg')

    print(f'pandas: v{pd.__version__}, spaCy: v{spacy.__version__}')

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s_suff, p_flag = argparse_wrapper(arg_parser)
    suppress_W008()
    fix_dirs(s_suff, p_flag)
    print_pipeline(datasets, s_suff, p_flag)

    timer = Timer(first_process='reading original csv files')
    dataframes: list[pd.DataFrame] = load_dataframes(datasets, s_suff)

    dataframe = dataframes.pop(0)   # "train" is the fundamental dataframe

    for df in dataframes:
        dataframe = pd.merge(dataframe, df, how='left', on='product_uid')

    # code block between the separators is responsible for parsing relevant strings and storing this data
    # (if parsed data is already present on the disk, this step can be skipped)
    # ----------------------------------------------------------------------------------------------------
    if p_flag:
        parsable_cols = [col for col in dataframe.columns if (dataframe[col].dtype == object)]
        for col in parsable_cols:
            timer(f'parsing {col}')
            s: pd.Series = dataframe[col]
            db: spacy.tokens.DocBin = parse_data(s, nlp)
            timer('saving parsed data to disk')
            store_as_docbin(db, s.name, s_suff)
            del s, db   # help Python with garbage collection
        timer()         # print time it took for the last step
        quit()		    # when data has been parsed and stored, prematurely quit the script
    # ----------------------------------------------------------------------------------------------------++
    
    timer('reading search_term spacy docs')
    docs: list[spacy.tokens.Doc] = load_docs('search_term', nlp, s_suff)
    dataframe['search_term'] = docs
    
    parsable_cols = [col for col in dataframe.columns if (dataframe[col].dtype == object and col != 'search_term')]
    
    for col in parsable_cols:
        timer(f'reading {col} spacy docs')
        docs: list[spacy.tokens.Doc] = load_docs(col, nlp, s_suff)
        dataframe[col] = docs
        
        timer(f'calculating similarity search_term <-> {col}')
        dataframe[f'zipped_{col}'] = tuple(zip(dataframe['search_term'], dataframe[col]))
        dataframe[f'sim_{col}'] = dataframe[f'zipped_{col}'].map(lambda x: x[0].similarity(x[1]))
        dataframe.drop([col, 'zipped_'+col], axis=1, inplace=True)

    timer('gathering results')
    temp_t, temp_d  = {}, {}
    for _, row in dataframe.iterrows():
        rel, sim_t, sim_d = row['relevance'], row['sim_product_title'], row['sim_product_description'] 
        try: temp_t[rel].append(sim_t)
        except KeyError: temp_t.update({rel: [sim_t]})
        try: temp_d[rel].append(sim_d)
        except KeyError: temp_d.update({rel: [sim_d]})

    timer()

    res_t, res_d = {}, {}
    for rel, sim_list in temp_t.items():
        res_t.update({rel: sum(sim_list)/len(sim_list)})
    for rel, sim_list in temp_d.items():
        res_d.update({rel: sum(sim_list)/len(sim_list)})
    
    print(' rel | title | descr ')
    print('-----+-------+-------')
    for rel in sorted(res_t.keys()):
        print(f'{rel:<4} | {round(res_t[rel], 3):<5} | {round(res_d[rel], 3):<5}')

if __name__ == "__main__":
    main()
