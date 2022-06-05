"""
Main
===
Script that should be run to generate all results.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------
import argparse			# specifying args from command line |
# dependencies ---------------------------------------------
import pandas as pd		# dataframes						|
import spacy			# natural language processing		|
# local imports -----------------------------------------------------------------------------
from helper import argparse_wrapper, suppress_W008, fix_dirs, Timer		# general utilities  |
from datamanager import (load_dataframes, parse_dataframes,				# data management	 |
						 store_docs_as_docbins, create_doc_dataframes)	# ""				 |
from processing import calc_similarity_scores							# further processing |
# -------------------------------------------------------------------------------------------

def main():

	print(f'pandas: v{pd.__version__}, spaCy: v{spacy.__version__}')

	full, parse = argparse_wrapper(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter))
	suppress_W008()
	fix_dirs()

	colnames = ['product_title', 'search_term']
	nlp: spacy.Language = spacy.load('en_core_web_lg')

	timer = Timer(first_process='reading csv files into dataframes')
	dataframes: dict[str, pd.DataFrame] = load_dataframes(['train', 'test'], full=full)

	# code block between the separators is responsible for parsing relevant strings and storing this data
	# (if parsed data is already present on the disk, this step can be skipped)
	# ----------------------------------------------------------------------------------------------------
	if parse:
		timer('parsing strings to docs')
		dataframes: dict[str, pd.DataFrame] = parse_dataframes(dataframes, colnames, nlp)

		timer('storing docs as docbins')
		store_docs_as_docbins(dataframes, colnames, full=full)

		timer()
		quit()		# when data has been parsed and stored, prematurely quit the script
	# ----------------------------------------------------------------------------------------------------

	timer('reading docbins into dataframes')
	dataframes: dict[str, pd.DataFrame] = create_doc_dataframes(dataframes, colnames, nlp, full=full)

	# from this point, we start using processing functions

	timer('calculating similarity scores')
	dataframes: dict[str, pd.DataFrame] = calc_similarity_scores(dataframes, ['product_title'])

	timer()

	for df_name, dataframe in dataframes.items():
		print()
		print(df_name)
		print(dataframe.columns)


if __name__ == "__main__":
	main()
