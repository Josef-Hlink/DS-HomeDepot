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
from helper import load_dataframes, parse_dataframes, store_as_docs, create_doc_dfs

def main():
	full: bool = parse_wrapper(argparse.ArgumentParser())
	fix_dirs()
	colnames = ['product_title', 'search_term']

	timer = Timer(first_process='reading csv files')
	dataframes: dict[str, pd.DataFrame] = load_dataframes(['train', 'test'], full=full)

	timer('parsing')
	parsed_dataframes = parse_dataframes(dataframes, colnames)

	timer('storing as docs')
	store_as_docs(parsed_dataframes, colnames)

	timer('reading docs back in')
	loaded_dataframes: dict[str, pd.DataFrame] = create_doc_dfs(dataframes, colnames)

	timer()

	for name, dataframe in loaded_dataframes.items():
		print(name)
		print(dataframe.head(5))
		print(dataframe.columns)
		for index, row in dataframe.head(5).iterrows():
			# little bit of testing
			print(type(row['search_term']))
			print(row['search_term'], '<->', row['product_title'], row['search_term'].similarity(row['product_title']))

if __name__ == "__main__":
	main()
