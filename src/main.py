"""
Main
===
Script that should be run to obtain all results.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library
import argparse
# dependencies
import pandas as pd
import spacy
# local imports
from helper import parse_wrapper, fix_dirs, Timer
from helper import load_dataframes, parse_dataframes, store_docs_as_docbins, create_doc_dataframes

def main():
	full: bool = parse_wrapper(argparse.ArgumentParser())
	fix_dirs()
	colnames = ['product_title', 'search_term']
	nlp: spacy.Language = spacy.load('en_core_web_lg')

	timer = Timer(first_process='reading csv files')
	dataframes: dict[str, pd.DataFrame] = load_dataframes(['train', 'test'], full=full)

	# ------------------------------------------------------------------------------------------------
	timer('parsing')
	dataframes: dict[str, pd.DataFrame] = parse_dataframes(dataframes, colnames, nlp)

	timer('storing as docs')
	store_docs_as_docbins(dataframes, colnames)

	# timer()
	# quit()
	# ------------------------------------------------------------------------------------------------

	timer('reading docs back in')
	dataframes: dict[str, pd.DataFrame] = create_doc_dataframes(dataframes, colnames, nlp)

	timer()

	for df_name, dataframe in dataframes.items():
		print()
		print(df_name)
		for _, row in dataframe.head(10).iterrows():
			# little bit of testing
			st, pt = row['search_term'], row['product_title']
			print(f'{round(st.similarity(pt), 3):<5}', '|', st, '<->', pt)

if __name__ == "__main__":
	main()
