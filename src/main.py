"""
Main
---
Data Science Assignment 3 - Home Depot Search Results
"""

import numpy as np
import pandas as pd
import spacy
from helper import fix_dirs, load_data, Timer

def main():
	fix_dirs()
	test()

def test():

	timer = Timer(first_process='reading in train, test & descriptions')

	train, test, descriptions = load_data(['train', 'test', 'product_descriptions'])
	
	timer(next_process='reading in attributes')
	
	attributes = load_data(['attributes'])[0]

	timer()

	print('\ntest')
	print(test.head(1))
	print('\ntrain')
	print(train.head(1))
	print('\ndescriptions')
	print(descriptions.head(1))
	print('\nattributes')
	print(attributes.head(1))


if __name__ == "__main__":
	main()
