"""
Helper
===
Complex functions that are called from main.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library
import argparse
import os, sys, re, shutil		# directories
from datetime import datetime	# printing experiment starting time
import time						# getting time indications during the experiment
from functools import lru_cache	# speeding up nlp parsing
# dependencies
import pandas as pd				# reading in data
import spacy

BOLD = lambda string: f'\033[1m{string}\033[0m'

def argparse_wrapper(parser: argparse.ArgumentParser) -> bool:
	"""Returns the parsed arguments of the file"""
	parser.add_argument('-f', '--full', action='store_true',
						help=('run script on full dataset, default is to run on sample data'))
	
	return parser.parse_args().full

def fix_dirs() -> None:
	"""Changes cwd to src, and creates the necessary directories"""
	cwd = os.getcwd()
	if cwd.split(os.sep)[-1] != 'src':
		if not os.path.exists(os.path.join(cwd, 'src')):
			print(f'Please work from either the parent directory "{BOLD("Home-Depot")}",',
				  f'or from "{BOLD("src")}" in order to run any scripts that are in "src".')
			sys.exit(1)
		os.chdir(os.path.join(cwd, 'src'))
		cwd = os.getcwd()
		caller = re.search(r'src(.*?).py', str(sys._getframe(1))).group(1)[1:] + '.py'
		print(f'\n WARNING: Working directory changed to "{cwd}".',
			  f'Consider running {BOLD(caller)} directly from "src" dir next time.\n')
	
	if not os.path.exists(results_dir := os.path.join(cwd, '..', 'results')):
		os.mkdir(results_dir)

class Timer:
	def __init__(self, first_process: str) -> None:
		"""Sets up a timer object and prints the name of the first process"""
		print(f'\nexperiment started at {datetime.now().strftime("%H:%M:%S")}')
		print(f'\n{first_process}: ', end='')
		self.tic: float = time.perf_counter()

	def __call__(self, next_process: str = None) -> None:
		"""Prints the time it took to complete the previous process and if specified, the name of the next process"""
		passed: float = round((toc := time.perf_counter()) - self.tic, 6)
		print(f'{BOLD(passed)} s', end='')
		if next_process is not None:
			print(f'\n{next_process}: ', end='')
			self.tic = toc
		else:
			print()

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
			df.dropna(inplace=True)
			dataframes.update({filename: df})
		except FileNotFoundError:
			print(f'No file called {BOLD(filename)} is present in the data directory.')
			sys.exit(1)
	
	return dataframes

def parse_dataframes(dataframes: dict[str, pd.DataFrame], colnames: list[str],
					 nlp: spacy.Language) -> dict[str, pd.DataFrame]:
	"""
	In the given dataframes, converts the string values in the given dataframes into spaCy `doc` objects.
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
		"""small wrapper so caching can provide ±30% speed improvement"""
		return nlp(string)

	for df_name, df in dataframes.items():
		for col in colnames:
			try: df[col] = df[col].apply(_nlp_wrapper)
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
		shutil.rmtree(db_dir)
	os.mkdir(db_dir)

	doc_bins = {df_name: {col: spacy.tokens.DocBin(docs=df[col]) \
				for col in colnames} for df_name, df in dataframes.items()}
	for df_name, col_dict in doc_bins.items():
		if not os.path.exists((df_dir := os.path.join(db_dir,df_name))):
			os.mkdir(df_dir)
		for col, doc_bin in col_dict.items():
			doc_bin.to_disk(os.path.join(df_dir,col+'.spacy'))

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
	doc_bins = {df_name: {col: spacy.tokens.DocBin().from_disk(os.path.join(db_dir,df_name,col+'.spacy')) \
				for col in colnames} for df_name in dataframes.keys()}

	for df_name, df in dataframes.items():
		for col in colnames:
			docs = list(doc_bins[df_name][col].get_docs(nlp.vocab))
			df[col] = docs
	
	return dataframes
