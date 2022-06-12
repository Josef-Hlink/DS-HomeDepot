"""
Helper
===
General helper functions used throughout many scripts.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# python standard library --------------------------------------------------------
import argparse                 # easier switching between sample & full datasets |
import warnings                 # suppressing specific warning                    |
import os, sys, re              # directories                                     |
from datetime import datetime   # printing experiment starting time               |
import time                     # getting time indications during the experiment  |
# --------------------------------------------------------------------------------

BOLD = lambda string: f'\033[1m{string}\033[0m'
PATH = lambda *args: os.path.join(os.getcwd(), *args)

def argparse_wrapper(parser: argparse.ArgumentParser) -> tuple[str, bool, bool, bool, bool]:
    """
    Returns the parsed arguments of the file.
    ### params
        - the argparse `ArgumentParser` object that was instantiated in main.py
    
    ### returns
        - s_suff: determines if the sample dataset should be used, this will be included at all the right places
        - p_flag: toggles parsing of the string data into spaCy `Doc` objects
        - c_flag: toggles calculating all metrics and storing this as NumPy arrays
        - d_flag: toggles distribution plotting
        - t_flag: toggles training and testing
    """
    parser.add_argument('-f', '--full', action='store_true',
                        help='run script on full dataset, default is to run on sample data')
    parser.add_argument('-p', '--parse', action='store_true',
                        help='parse string data into spaCy docs, required for first run!')
    parser.add_argument('-c', '--calc_sim', action='store_true',
                        help='calculate similarity scores, parsed data needs to be present on disk!')
    parser.add_argument('-d', '--dis_plots', action='store_true',
                        help='create and store distribution plots, similarity data needs to be present on disk!')
    parser.add_argument('-t', '--train_test', action='store_true',
                        help='train and test a RF regression model on all numerical data that is present on disk')
    
    s_suff = '' if parser.parse_args().full else '_sample'
    p_flag = parser.parse_args().parse
    c_flag = parser.parse_args().calc_sim
    d_flag = parser.parse_args().dis_plots
    t_flag = parser.parse_args().train_test
    return (s_suff, p_flag, c_flag, d_flag, t_flag)

def suppress_W008() -> None:
    """Suppresses useless warning that (correctly) states some of the words in the data are not recognized by spaCy"""
    warnings.filterwarnings('ignore', message=r'\[W008\]', category=UserWarning)

def fix_dirs(s_suff: str) -> None:
    """Changes cwd to src, and creates the necessary storage and results directories"""
    cwd = os.getcwd()
    if cwd.split(os.sep)[-1] != 'src':
        if not os.path.exists(PATH('src')):
            print(f'\nERROR: Please work from either the parent directory "{BOLD("Home-Depot")}",',
                  f'or from "{BOLD("src")}" in order to run any scripts that are in "src".\n')
            sys.exit(1)
        os.chdir(PATH('src'))
        cwd = os.getcwd()
        caller = re.search(r'src(.*?).py', str(sys._getframe(1))).group(1)[1:] + '.py'
        print(f'\n WARNING: Working directory changed to "{cwd}".',
              f'Consider running {BOLD(caller)} directly from "src" dir next time.\n')
    
    global _S
    _S = s_suff
    if not os.path.exists(storage_dir := PATH('..',f'storage{_S}')):
        os.mkdir(storage_dir)
    if not os.path.exists(results_dir := PATH('..',f'results{_S}')):
        os.mkdir(results_dir)

def print_pipeline(datasets: list[str], p_flag: bool, c_flag: bool, d_flag: bool, t_flag: bool) -> None:
    """Prints how the pipeline will be executed based on the datasets and the flags provided by the user"""
    relevant_columns = {'train': ['product_title', 'search_term'],
                        'product_descriptions': ['product_description']}
    
    datasets_to_read = ', '.join([ds+_S+'.csv' for ds in datasets])
    columns_to_parse = ', '.join(set(col for dataset in datasets for col in relevant_columns[dataset]))
    columns_to_calc = columns_to_parse.replace('search_term, ', '').replace(', search_term', '')
    columns_to_plot = columns_to_calc

    pipeline: list[str] = [f'read {datasets_to_read}']

    if p_flag:
        pipeline += [f'parse {columns_to_parse} data into spaCy docs', 'store spaCy doc data to disk']
    if c_flag:
        pipeline += [f'load stored spaCy doc data into columns {columns_to_parse}',
                     'calculate length of search_term'
                     f'calculate similarity scores for {columns_to_calc}',
                     'store scores to disk as arrays']
    if d_flag:
        pipeline += [f'load stored scores into columns {columns_to_plot}', 'plot data distributions',
                     'save distribution plots to disk']
    if t_flag:
        pipeline += ['train Random Forest Regressor on numerical data',
                     'test accuracy of the Random Forest Regressor on separate test set']
    
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
        passed: float = round((toc := time.perf_counter()) - self.tic, 2)
        print(f'{BOLD(passed)} s', end='')
        if next_process is not None:
            print(f'\n{next_process}: ', end='')
            self.tic = toc
        else:
            print()
            total_time: float = time.perf_counter() - self.start
            minutes, seconds = int((total_time) // 60), round((total_time) % 60, 1)
            total_time_string = f'{minutes}:{str(seconds).zfill(4)} min' if minutes else f'{seconds} sec\n'
            print(f'\nexperiment took {total_time_string}')
