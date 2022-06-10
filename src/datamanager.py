"""
Data Manager
===
Functions pertaining to data loading and parsing.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------------
import os, sys, shutil              # directories                 |
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
    global _S
    _S = s_suff
    data_dir: str = PATH('..',f'data{_S}')
    
    dataframes: list[pd.DataFrame] = []

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

def create(dir_name: str) -> None:
    """Makes a directory in the storage folder"""
    if os.path.exists(dir := PATH('..','storage'+_S,dir_name)):
        shutil.rmtree(dir)      # clear old data
    os.mkdir(dir)               # make fresh directory, also works if no directory was present in the first place

def require(dir_name: str) -> None:
    """Checks if docbins or arrays directory exists"""
    if not os.path.exists((dir := PATH('..',f'storage{_S}',dir_name))):
        print(f'\nERROR: Some data is missing, please verify that all necessary data for the task is',
                f'present in the storage{_S}/{BOLD(dir)} directory.\n')
        sys.exit(1)

def store_as_docbin(db: spacy.tokens.DocBin, col_name: str) -> None:
    """
    Stores a spaCy `DocBin`on the user's disk at the specified location in the .spacy file format.
    ### params
        - db: a spaCy `DocBin` that is to be saved
        - col_name: name of the column in that `DataFrame` represented by the given `DocBin`
    """
    loc = PATH('..',f'storage{_S}','docbins',f'{col_name}.spacy')
    db.to_disk(loc)	    # store DocBin to disk at specified location

def load_docs(col_name: str, nlp: spacy.Language) -> pd.DataFrame:
    """
    For a given column, loads the spaCy `Doc` objects present on the user's disk.
    ### params
        - col_name: name of the column to be loaded
        - nlp: the spaCy `Language` object used to parse the strings
        - s_suff: determines whether the experiment is run on sample dataset
    ### returns
        - the `Doc` data that was present on the disk
    """
    db = spacy.tokens.DocBin().from_disk(PATH('..',f'storage{_S}','docbins',f'{col_name}.spacy'))
    docs = list(db.get_docs(nlp.vocab))	    # extract all Docs from DocBin
    return docs

def store_as_array(columns: tuple[pd.Series, pd.Series]) -> None:
    """Stores numerical data from two columns as a 2D NumPy array in the .npy file format"""
    relevance, similarity = columns[0], columns[1]
    array = np.array(list(zip(relevance.values, similarity.values)))
    loc = PATH('..',f'storage{_S}','arrays', similarity.name+'.npy')
    np.save(loc, array)	    # store array to disk at specified location

def load_array(col_name: str) -> np.ndarray:
    """For a given name, loads the NumPy array present on the user's disk"""
    return np.load(PATH('..',f'storage{_S}','arrays',f'{col_name}.npy'))
