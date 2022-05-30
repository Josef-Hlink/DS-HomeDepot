"""
Helper
---
Data Science Assignment 3 - Home Depot Search Results
"""

import os, sys, re				# directories
from datetime import datetime	# printing experiment starting time
import time						# getting time indications during the experiment
import pandas as pd				# reading in data

BOLD = lambda string: f'\033[1m{string}\033[0m'

def fix_dirs() -> None:
	"""Changes cwd to src, and creates a results dir on the same level if not already present"""
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

def load_data(filenames: list[str], sample: bool = False) -> list[pd.DataFrame]:
	"""Loads given csv files into a list as pandas DataFrames, don't put .csv in the filename"""
	if sample:
		data_dir: str = os.path.join(os.getcwd(), '..', 'sample_data')
	else:
		data_dir: str = os.path.join(os.getcwd(), '..', 'data')
	
	files: list[pd.DataFrame] = []
	for filename in filenames:
		try:
			filepath = os.path.join(data_dir, filename+'.csv')
			files.append(pd.read_csv(filepath, encoding='ISO-8859-1'))
		except FileNotFoundError:
			print(f'No file called {BOLD(filename)} is present in the data directory.')
			sys.exit(1)
	return files

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
