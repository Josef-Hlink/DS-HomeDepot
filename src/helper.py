"""
Helper
===
General helper functions that are called from main.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library --------------------------------------------------------
import argparse                 # easier switching between sample & full datasets |
import warnings                 # suppressing specific warning                    |
import os, sys, re, shutil      # directories                                     |
from datetime import datetime   # printing experiment starting time               |
import time                     # getting time indications during the experiment  |
# --------------------------------------------------------------------------------

BOLD = lambda string: f'\033[1m{string}\033[0m'

def argparse_wrapper(parser: argparse.ArgumentParser) -> tuple[str, bool]:
    """Returns the parsed arguments of the file"""
    parser.add_argument('-f', '--full', action='store_true',
                        help='run script on full dataset, default is to run on sample data')
    parser.add_argument('-p', '--parse', action='store_true',
                        help='parse string data into spaCy docs, required for first run!')
    
    s_suff = '' if parser.parse_args().full else '_sample'
    p_flag = parser.parse_args().parse
    return (s_suff, p_flag)

def suppress_W008() -> None:
    """Suppresses useless warning that (correctly) states some of the words in the data are not recognized by spaCy"""
    warnings.filterwarnings('ignore', message=r'\[W008\]', category=UserWarning)

def fix_dirs(s_suff: str, p_flag: bool) -> None:
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
    
    if os.path.exists((db_dir := os.path.join('..','docbins'+s_suff))):
        if p_flag:
            shutil.rmtree(db_dir)    # clear old data
            os.mkdir(db_dir)         # make fresh directory in case we want to create a new one
    else:
        os.mkdir(db_dir)             # make fresh directory in case there wasn't one to start with
    if not os.path.exists(results_dir := os.path.join(cwd, '..', 'results')):
        os.mkdir(results_dir)

def print_pipeline(datasets: list[str], s_suff: str, p_flag: bool) -> None:
    lookup_table = {'train': ['product_title', 'search_term'],
                    'product_descriptions': ['product_description'],
                    'attributes': ['attributes']}
    
    datasets_to_read = ', '.join([ds+s_suff+'.csv' for ds in datasets])
    columns_to_parse = ', '.join(set(col for dataset in datasets for col in lookup_table[dataset]))

    if p_flag:
        pipeline = [f'read {datasets_to_read}',
                    f'parse {columns_to_parse} data into spaCy docs',
                    'store spaCy doc data to disk']
    else:
        pipeline = [f'read {datasets_to_read}',
                    f'load stored spaCy doc data into columns {columns_to_parse}',
                    'calculate similarity scores']
    print('\npipeline:')
    for pipe in pipeline: print('*', pipe)

class Timer:
    def __init__(self, first_process: str) -> None:
        """Sets up a timer object and prints the name of the first process"""
        print(f'\nexperiment started at {datetime.now().strftime("%H:%M:%S")}')
        print(f'\n{first_process}: ', end='')
        self.start: float = time.perf_counter()
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
            total_time: float = time.perf_counter() - self.start
            minutes = int((total_time) // 60)
            seconds = round((total_time) % 60, 1)
            total_time_string = f'{minutes}:{str(seconds).zfill(4)} min' if minutes else f'{seconds} sec\n'
            print(f'\nexperiment took {total_time_string}')
