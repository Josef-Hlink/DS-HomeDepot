"""
Data Manager
===
Functions pertaining to data loading and parsing.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------------
import os, sys                      # directories                 |
from functools import lru_cache     # speeding up nlp parsing     |
# dependencies ---------------------------------------------------
import pandas as pd                 # dataframes                  |
import spacy                        # natural language processing |
import numpy as np                  # arrays                      |
# local imports --------------------------------------------------
from helper import BOLD, PATH       # TUI, directories            |
# ----------------------------------------------------------------

def load_dataframes(filenames: list[str], s_suff: str) -> list[pd.DataFrame]:
    """
    Loads given csv files into a list as pandas `DataFrame`s.
    ### params
        - filenames: names of the files to load, path and file extension do not need to be specified
        - s_suff: determines whether the experiment is run on sample dataset
    ### returns
        - dataframes: list containing all of the loaded `DataFrame`s
    """
    dataframes: list[pd.DataFrame] = []
    data_dir: str = PATH('..','sample_data') if len(s_suff) else PATH('..','data')

    for filename in filenames:
        try:
            filepath = os.path.join(data_dir, filename+'.csv')
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
            df.dropna(inplace=True)         # remove all corrupted entries
            dataframes.append(df)           # add the loaded dataframe to the list
        except FileNotFoundError:
            print(f'Error: No file called {BOLD(filename)} is present in the data directory.')
            sys.exit(1)
    
    return dataframes

def parse_data(s: pd.Series, nlp: spacy.Language) -> spacy.tokens.DocBin:
    """
    In the given pandas `Series`, converts the string values into spaCy `Doc` objects.
    ### params
        - s: a pandas `Series` object containing strings
        - nlp: the spaCy `Language` object used to parse the strings
    ### returns
        - db: a spaCy `DocBin` object containing all the parsed string data
    """

    @lru_cache(maxsize=1)
    def _nlp_wrapper(string: str) -> spacy.tokens.Doc:
        """small wrapper for speed improvement on product title data"""
        return nlp(string)

    s = s.apply(_nlp_wrapper) if s.name == 'product_title' else s.apply(nlp)
    
    db = spacy.tokens.DocBin(docs=s)    # store as spaCy DocBin
    return db

def store_as_docbin(db: spacy.tokens.DocBin, col_name: str, s_suff: str) -> None:
    """
    Stores a spaCy `DocBin`on the user's disk at the specified location in the .spacy file format.
    ### params
        - db: a spaCy `DocBin` that is to be saved
        - col_name: name of the column in that `DataFrame` represented by the given `DocBin`
        - s_suff: determines whether the experiment is run on sample dataset
    """
    loc = PATH('..',f'docbins{s_suff}',f'{col_name}.spacy')
    db.to_disk(loc)	    # store DocBin to disk at specified location

def load_docs(col_name: str, nlp: spacy.Language, s_suff: str) -> pd.DataFrame:
    """
    For a given column, loads the spaCy `Doc` objects present on the user's disk.
    ### params
        - col_name: name of the column to be loaded
        - nlp: the spaCy `Language` object used to parse the strings
        - s_suff: determines whether the experiment is run on sample dataset
    ### returns
        - the `Doc` data that was present on the disk
    """
    db = spacy.tokens.DocBin().from_disk(PATH('..',f'docbins{s_suff}',f'{col_name}.spacy'))
    docs = list(db.get_docs(nlp.vocab))	    # extract all Docs from DocBin
    return docs

def store_as_array(columns: tuple[pd.Series, pd.Series], s_suff: str) -> None:
    """
    Stores numerical data from two columns as a 2D NumPy array in the .npy file format.
    """
    relevance, similarity = columns[0], columns[1]
    array = np.array(list(zip(relevance.values, similarity.values)))
    loc = PATH('..','arrays'+s_suff, similarity.name+'.npy')
    np.save(loc, array)	    # store array to disk at specified location

def load_array(col_name: str, s_suff: str) -> np.ndarray:
    """
    For a given column, loads the NumPy array present on the user's disk
    """
    return np.load(PATH('..','arrays'+s_suff, 'sim_'+col_name+'.npy'))