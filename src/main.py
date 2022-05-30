"""
Main
---
Data Science Assignment 3 - Home Depot Search Results
"""

import numpy as np
import pandas as pd
import spacy
from helper import fix_dirs, load_data

def main():
	fix_dirs()
	data = load_data(['test.csv'])
	test = data[0]
	print(test.readlines()[0])

if __name__ == "__main__":
	main()
