"""
Helper
===
General helper functions that are called from main.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library --------------------------------------------------------
import argparse					# easier switching between sample & full datasets |
import os, sys, re				# directories									  |
from datetime import datetime	# printing experiment starting time				  |
import time						# getting time indications during the experiment  |
# --------------------------------------------------------------------------------

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
