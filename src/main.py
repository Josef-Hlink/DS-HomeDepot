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
from helper import (argparse_wrapper, suppress_W008,					# general utilities	 |
					fix_dirs, print_pipeline, Timer)					# ""				 |
from datamanager import (load_dataframes, parse_dataframes,				# data management	 |
						 store_docs_as_docbins, create_doc_dataframes)	# ""				 |
from processing import calc_similarity_scores							# further processing |
# -------------------------------------------------------------------------------------------

def main():

	print(f'pandas: v{pd.__version__}, spaCy: v{spacy.__version__}')

	full, parse = argparse_wrapper(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter))
	suppress_W008()
	fix_dirs()

	datasets = ['train', 'test', 'product_descriptions']
	colnames = {'train': ['product_title', 'search_term'],
				'test':  ['product_title', 'search_term'],
				'product_descriptions': ['product_description']}

	print_pipeline(datasets, colnames, full, parse)
	nlp: spacy.Language = spacy.load('en_core_web_lg')

	timer = Timer(first_process='reading csv files into dataframes')
	dataframes: dict[str, pd.DataFrame] = load_dataframes(datasets, full=full)

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

	# test is temporarily disregarded altogether, and train is merged with product_descriptions
	# this is messy, very non-modular code, just for testing

	train_df, prod_desc_df = dataframes['train'], dataframes['product_descriptions']

	complete_df = pd.merge(train_df, prod_desc_df, how='left', on='product_uid')

	dataframes = {'complete': complete_df}
	colnames = {'complete': ['product_title', 'product_description']}

	timer('calculating similarity scores')
	dataframes: dict[str, pd.DataFrame] = calc_similarity_scores(dataframes, colnames)
	df_name = 'complete'
	dataframe = dataframes[df_name]

	timer('gathering results')
	temp_t, temp_d  = {}, {}
	for _, row in dataframe.iterrows():
		rel, sim_t, sim_d = row['relevance'], row['sim_product_title'], row['sim_product_description'] 
		# print(f'{rel:<4} | {round(sim_t, 3):<5} | {round(sim_d, 3):<5}')
		try: temp_t[rel].append(sim_t)
		except KeyError: temp_t.update({rel: [sim_t]})
		try: temp_d[rel].append(sim_d)
		except KeyError: temp_d.update({rel: [sim_d]})

	timer()

	res_t, res_d = {}, {}
	for rel, sim_list in temp_t.items():
		res_t.update({rel: sum(sim_list)/len(sim_list)})
	for rel, sim_list in temp_d.items():
		res_d.update({rel: sum(sim_list)/len(sim_list)})
	
	print(dataframe.columns)
	print()
	print(' rel | title | descr ')
	print('-----+-------+-------')
	for rel in sorted(res_t.keys()):
		print(f'{rel:<4} | {round(res_t[rel], 3):<5} | {round(res_d[rel], 3):<5}')

if __name__ == "__main__":
	main()
