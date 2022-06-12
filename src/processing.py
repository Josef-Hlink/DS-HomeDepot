"""
Processing
===
Functions pertaining to processing data.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------
from functools import lru_cache     # speeding up parsing   |
# dependencies ---------------------------------------------
import pandas as pd                 # dataframes            |
import spacy                        # NLP                   |
# ----------------------------------------------------------

def parse_data(series: pd.Series, nlp: spacy.Language) -> spacy.tokens.DocBin:
    """
    In the given pandas `Series`, converts the string values into spaCy `Doc` objects.
    ### params
        - series: a pandas `Series` object containing strings
        - nlp: the spaCy `Language` object used to parse the strings
    ### returns
        - docbin: a spaCy `DocBin` object containing all the parsed string data
    """

    @lru_cache(maxsize=1)
    def _nlp_wrapper(string: str) -> spacy.tokens.Doc:
        """small wrapper for speed improvement on product title & product description data"""
        return nlp(string)

    series = series.apply(_nlp_wrapper)
    
    docbin = spacy.tokens.DocBin(docs=series)       # store as spaCy DocBin
    return docbin

def calc_semantic_similarity(series: pd.Series) -> pd.Series:
    """Calculates the semantic similarity between two spaCy `Doc` objects"""
    return series.map(lambda docs: docs[0].similarity(docs[1]))

def calc_simple_similarity(series: pd.Series) -> pd.Series:
    """Calculates the number of "hits" between two (stemmed) spaCy `Doc` objects"""
    return series.map(lambda docs: sum(int(' '.join(token.lemma_ for token in docs[1]).find(word)>=0) \
                                                    for word in [token.lemma_ for token in docs[0]]))

