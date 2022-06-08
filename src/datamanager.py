"""
Data Manager
===
Functions pertaining to data loading.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------------
import os, sys, shutil              # directories                 |
from functools import lru_cache     # speeding up nlp parsing     |
# dependencies ---------------------------------------------------
import pandas as pd                 # dataframes                  |
import spacy                        # natural language processing |
# local imports --------------------------------------------------
from helper import BOLD             # slightly improved TUI       |
# ----------------------------------------------------------------

def load_dataframes(filenames: list[str], full: bool = False) -> dict[str, pd.DataFrame]:
    """
    Loads given csv files into a list as pandas DataFrames.
    ---
    ### params
        - filenames: names of the files to load, no need to specify path and file extension
        - full: set to True to run on the entire dataset
    
    ### returns
        - dataframes: dict with
            * keys: the names of the dataframes (just the specified filename)
            * values: the actual dataframes
    """
    dataframes: dict[str, pd.DataFrame] = {}
    if full: data_dir: str = os.path.join(os.getcwd(), '..', 'data')
    else: data_dir: str = os.path.join(os.getcwd(), '..', 'sample_data')

    for filename in filenames:
        try:
            filepath = os.path.join(data_dir, filename+'.csv')
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
            df.dropna(inplace=True)             # remove all corrupted entries
            dataframes.update({filename: df})   # add the loaded dataframe to the dict
        except FileNotFoundError:
            print(f'Error: No file called {BOLD(filename)} is present in the data directory.')
            sys.exit(1)
    
    return dataframes

def parse_data(s: pd.Series, nlp: spacy.Language) -> spacy.tokens.DocBin:
    """
    In the given pandas Series, converts the string values into spaCy `Doc` objects.
    ---
    ### params
        - s: a pandas Series object containing strings
        - nlp: the spaCy language object used to parse the strings
    
    ### returns
        - db: a spaCy `DocBin` containing all the parsed string data
    """

    @lru_cache(maxsize=50)
    def _nlp_wrapper(string: str) -> spacy.tokens.Doc:
        """small wrapper so caching can be used for a slight (but welcome) speed improvement"""
        return nlp(string)

    s = s.apply(_nlp_wrapper)           # parse all data
    db = spacy.tokens.DocBin(docs=s)    # store as spaCy DocBin
    return db

def store_as_docbin(db: spacy.tokens.DocBin, db_dir: str, df_name: str, col: str) -> None:
    """
    Stores a spaCy `DocBin`on the user's disk at the specified location
    ---
    ### params
        - db: a spaCy `DocBin` that is to be saved
        - db_dir: the location of all `DocBin`s
        - df_name: name of the dataframe where the `DocBin` originated from
        - col: name of the column in that dataframe represented by the given `DocBin`
    """

    if not os.path.exists((df_dir := os.path.join(db_dir,df_name))):
        os.mkdir(df_dir)
    db.to_disk(os.path.join(df_dir,col+'.spacy'))	# store DocBin to disk at specified location

def create_doc_dataframes(dataframes: dict[str, pd.DataFrame], colnames: dict[str, list[str]],
                          nlp: spacy.Language, full: bool = False) -> dict[str, pd.DataFrame]:
    """
    Replaces the string data in the given dataframes with corresponding spaCy `Doc` objects present on the user's disk
    ---
    ### params
        - dataframes: dict with 
            * keys: the names of the dataframes that should be modified
            * values: the actual dataframes
        - colnames: list of column names of which the string data should be replaced with corresponding `doc` objects
        - nlp: the spaCy language object used to parse the strings
        - full: set to True to run on the entire dataset
    
    ### returns
        - dataframes: the same dict, but with parsed dataframes
    """
    flag = '_sample' if not full else ''
    db_dir = os.path.join(os.getcwd(),'..','docbins'+flag)
    
    # load DocBins into dict from disk
    doc_bins = {df_name: {col: spacy.tokens.DocBin().from_disk(os.path.join(db_dir,df_name,col+'.spacy')) \
                for col in colnames[df_name]} for df_name in dataframes.keys()}
    # example where dataframes = ['train', 'test'] and col_names = ['search_term', 'product_title']:
    # doc_bins = {'train': {'search_term': <DocBin>, 'product_title': <DocBin>},
    #             'test':  {'search_term': <DocBin>, 'product_title': <DocBin>}}

    for df_name, df in dataframes.items():
        for col in colnames[df_name]:
            doc_bin = doc_bins[df_name][col]			# access DocBin in dict
            docs = list(doc_bin.get_docs(nlp.vocab))	# extract all Docs from DocBin
            df[col] = docs								# load into dataframe, overwriting string data
    
    return dataframes
