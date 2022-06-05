"""
Processing
===
Functions pertaining to further processing parsed data.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library ----------------------------------
import sys					# exiting on errors				|
# dependencies ---------------------------------------------
import pandas as pd			# dataframes					|
import spacy				# natural language processing	|
# local imports --------------------------------------------
from helper import BOLD		# slightly improved TUI			|
# ----------------------------------------------------------

def calc_similarity_scores(dataframes: dict[str, pd.DataFrame], col_names: list[str]) -> dict[str, pd.DataFrame]:
	
	def _calc_sim(doc1: spacy.tokens.Doc, doc2: spacy.tokens.Doc) -> float:
		"""Note: this would generate a warning, as some of the words in de dataset are not recognized by spaCy"""
		return doc1.similarity(doc2)
	
	for df_name, df in dataframes.items():
		for col in col_names:
			
			try: df[f'zipped_{col}'] = tuple(zip(df['search_term'], df[col]))
			except KeyError:
				print(f'Error: Dataframe "{BOLD(df_name)}" has no column called "{BOLD(col)}"')
				sys.exit(1)
			
			df[f'sim_{col}'] = df[f'zipped_{col}'].map(lambda x: _calc_sim(x[0], x[1]))
			df.drop(f'zipped_{col}', axis=1, inplace=True)
	
	return dataframes
