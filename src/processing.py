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

def calc_length(series: pd.Series) -> pd.Series:
    """Calculates the amount of words in an entry, note that a double space does register as a word"""
    return series.apply(lambda x: len(x))

def filter_low_similarities(dataframe: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Filters out entries with unworkably low similarity scores"""
    dataframe = dataframe[getattr(dataframe, col_name) > 0.1]
    return dataframe

def filter_rare_relevancies(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filters out entries with relevance scores that occur less than 5 times"""
    rel = dataframe.relevance
    dataframe = dataframe[(rel != 1.25) & (rel != 1.5) & (rel != 2.5)  & (rel != 2.75)]
    return dataframe
