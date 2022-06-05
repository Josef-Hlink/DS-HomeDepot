"""
Data Manager
===
Functions pertaining to data loading.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------------
import os, sys, shutil  		    # directories                 |
from functools import lru_cache	    # speeding up nlp parsing     |
# dependencies ---------------------------------------------------
import pandas as pd				    # dataframes                  |
import spacy                        # natural language processing |
# local imports --------------------------------------------------
from helper import BOLD             # slightly improving TUI      |
# ----------------------------------------------------------------

def load_dataframes(filenames: list[str], full: bool = False) -> dict[str, pd.DataFrame]:
	"""
	Loads given csv files into a list as pandas DataFrames.
	---
	### params
		- filenames: names of the files to load, no need to specify path and file extension
		- full: set to True to run on the entire dataset
	
	### returns
		- dataframes: dict with
			* keys: the names of the dataframes (just the specified filename)
			* values: the actual dataframes
	"""
	dataframes: dict[str, pd.DataFrame] = {}
	if full: data_dir: str = os.path.join(os.getcwd(), '..', 'data')
	else: data_dir: str = os.path.join(os.getcwd(), '..', 'sample_data')

	for filename in filenames:
		try:
			filepath = os.path.join(data_dir, filename+'.csv')
			df = pd.read_csv(filepath, encoding='ISO-8859-1')
			df.dropna(inplace=True)             # remove all corrupted entries
			dataframes.update({filename: df})   # add the loaded dataframe to the dict
		except FileNotFoundError:
			print(f'No file called {BOLD(filename)} is present in the data directory.')
			sys.exit(1)
	
	return dataframes

def parse_dataframes(dataframes: dict[str, pd.DataFrame], colnames: list[str],
					 nlp: spacy.Language) -> dict[str, pd.DataFrame]:
	"""
	In the given dataframes, converts the string values in the given dataframes into spaCy `Doc` objects.
	---
	### params
		- dataframes: dict with 
			* keys: the names of the dataframes on which the conversion should be done
			* values: the actual dataframes
		- colnames: list of column names containing parsable string data
		- nlp: the spaCy language object used to parse the strings
	
	### returns
		- dataframes: the same dict, but with parsed dataframes
	"""

	@lru_cache(maxsize=50)
	def _nlp_wrapper(string: str) -> spacy.tokens.Doc:
		"""small wrapper so caching can be used for a slight (but welcome) speed improvement"""
		return nlp(string)

	for df_name, df in dataframes.items():
		for col in colnames:
			try: df[col] = df[col].apply(_nlp_wrapper)  # this is where the parsing happens
			except KeyError:
				print(f'Dataframe "{BOLD(df_name)}" has no column called "{BOLD(col)}"')
				sys.exit(1)
	
	return dataframes

def store_docs_as_docbins(dataframes: dict[str, pd.DataFrame], colnames: list[str], full: bool = False) -> None:
	"""
	Stores the spaCy `Doc` objects present in the given dataframes on the user's disk as `DocBin` objects
	---
	### params
		- dataframes: dict with 
			* keys: the names of the dataframes containing `Doc` objects to be stored
			* values: the actual dataframes
		- colnames: list of column names containing `Doc` objects
		- full: set to True to run on the entire dataset
	"""
	flag = '_sample' if not full else ''
	if os.path.exists((db_dir := os.path.join('..','docbins'+flag))):
		shutil.rmtree(db_dir)   # clear old data
	os.mkdir(db_dir)            # make fresh directory

	# create DocBins for every relevant column in the dataframes and store in dict
	doc_bins = {df_name: {col: spacy.tokens.DocBin(docs=df[col]) \
				for col in colnames} for df_name, df in dataframes.items()}
	# example where dataframes = ['train', 'test'] and col_names = ['search_term', 'product_title']:
	# doc_bins = {'train': {'search_term': <DocBin>, 'product_title': <DocBin>},
	#             'test':  {'search_term': <DocBin>, 'product_title': <DocBin>}}

	for df_name, col_dict in doc_bins.items():
		if not os.path.exists((df_dir := os.path.join(db_dir,df_name))):
			os.mkdir(df_dir)
		for col, doc_bin in col_dict.items():
			doc_bin.to_disk(os.path.join(df_dir,col+'.spacy'))	# store DocBin to disk at specified location

def create_doc_dataframes(dataframes: dict[str, pd.DataFrame], colnames: list[str],
						  nlp: spacy.Language, full: bool = False) -> dict[str, pd.DataFrame]:
	"""
	Replaces the string data in the given dataframes with corresponding spaCy `Doc` objects present on the user's disk
	---
	### params
		- dataframes: dict with 
			* keys: the names of the dataframes that should be modified
			* values: the actual dataframes
		- colnames: list of column names of which the string data should be replaced with corresponding `doc` objects
		- nlp: the spaCy language object used to parse the strings
		- full: set to True to run on the entire dataset
	
	### returns
		- dataframes: the same dict, but with parsed dataframes
	"""
	flag = '_sample' if not full else ''
	db_dir = os.path.join(os.getcwd(),'..','docbins'+flag)
	
	# load DocBins into dict from disk
	doc_bins = {df_name: {col: spacy.tokens.DocBin().from_disk(os.path.join(db_dir,df_name,col+'.spacy')) \
				for col in colnames} for df_name in dataframes.keys()}
	# example where dataframes = ['train', 'test'] and col_names = ['search_term', 'product_title']:
	# doc_bins = {'train': {'search_term': <DocBin>, 'product_title': <DocBin>},
	#             'test':  {'search_term': <DocBin>, 'product_title': <DocBin>}}

	for df_name, df in dataframes.items():
		for col in colnames:
			doc_bin = doc_bins[df_name][col]			# access DocBin in dict
			docs = list(doc_bin.get_docs(nlp.vocab))	# extract all Docs from DocBin
			df[col] = docs								# load into dataframe, overwriting string data
	
	return dataframes
