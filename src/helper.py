"""
Helper
---
Data Science Assignment 3 - Home Depot Search Results
"""

from io import TextIOWrapper	# type hinting
import os, sys, re       		# directories

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

def load_data(filenames: list[str]) -> list[TextIOWrapper]:
	"""Loads given csv files into a list as TextIOWrappers"""
	data_dir: str = os.path.join(os.getcwd(), '..', 'data')
	files: list[TextIOWrapper] = []
	for filename in filenames:
		try: files.append(open(os.path.join(data_dir, filename), 'r', encoding='ISO-8859-1'))
		except FileNotFoundError:
			print(f'No file called {BOLD(filename)} is present in the data directory.')
			sys.exit(1)
	return files
