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
# local imports ----------------------------------------------------------------------------
from helper import argparse_wrapper, fix_dirs, Timer					# general utilities |
from datamanager import (load_dataframes, parse_dataframes,				# data management	|
						 store_docs_as_docbins, create_doc_dataframes)	# ""				|
# ------------------------------------------------------------------------------------------

def main():

	print(f'pandas: v{pd.__version__}, spaCy: v{spacy.__version__}')

	full, parse = argparse_wrapper(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter))
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
		quit()
	# ----------------------------------------------------------------------------------------------------

	timer('reading docbins into dataframes')
	dataframes: dict[str, pd.DataFrame] = create_doc_dataframes(dataframes, colnames, nlp, full=full)

	timer()

	for df_name, dataframe in dataframes.items():
		print()
		print(df_name)
		for _, row in dataframe.head(10).iterrows():
			# check if similarity functions work
			st, pt = row['search_term'], row['product_title']
			if df_name == 'train':
				print(f'{round(st.similarity(pt), 3):<5} | {row["relevance"]:<4} | {st} <-> {pt}')
			else:
				print(f'{round(st.similarity(pt), 3):<5} | {st} <-> {pt}')


if __name__ == "__main__":
	main()
