"""
Processing
===
Functions pertaining to further processing parsed data.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------
import sys                          # exiting on errors     |
from functools import lru_cache     # speeding up parsing   |
# dependencies ---------------------------------------------
import pandas as pd                 # dataframes            |
import spacy                        # NLP                   |
# local imports --------------------------------------------
from helper import BOLD             # TUI                   |
# ----------------------------------------------------------

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

def calc_semantic_similarity(series: pd.Series) -> pd.Series:
    return series.map(lambda docs: docs[0].similarity(docs[1]))

def calc_simple_similarity(series: pd.Series) -> pd.Series:
    return series.map(lambda docs: sum(int(' '.join(token.lemma_ for token in docs[1]).find(word)>=0) for word in [token.lemma_ for token in docs[0]]))

