"""
Model
===
Functions pertaining to training and testing.
---
Data Science Assignment 3 - Home Depot Search Results
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE

def train_and_test(dataframe: pd.DataFrame) -> float:

    y = dataframe['relevance'].values
    relevant_columns = ['sim_sim_product_title', 'sim_sim_product_description',
                        'sem_sim_product_title', 'sem_sim_product_description',
                        'len_of_query']
    X = dataframe[relevant_columns].values

    X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=40)

    RFR = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    BR = BaggingRegressor(RFR, n_estimators=45, max_samples=0.1, random_state=25)
    
    BR.fit(X_train, y_train)

    y_pred = BR.predict(X_test)
    RMSE = MSE(y_test, y_pred)**0.5

    return RMSE
