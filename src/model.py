"""
Model
===
Functions pertaining to training and testing.
---
Data Science Assignment 3 - Home Depot Search Results
"""

# dependencies ---------------------------------------------------------------------------------
import pandas as pd                                                     # dataframes            |
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor    # regression models     |
from sklearn.model_selection import train_test_split as TTS             # splitting data        |
from sklearn.metrics import mean_squared_error as MSE                   # measuring performance |
# local imports --------------------------------------------------------------------------------
from processing import filter_rare_relevancies, filter_low_similarities     # cleaning data     |
from plot import plot_feature_importances                                   # plotting features |
# ----------------------------------------------------------------------------------------------

def train_and_test(dataframe: pd.DataFrame) -> float:
    """Trains and tests a random forest regressor on the given dataframe and returns the root mean squared error"""
    dataframe = dataframe.copy()
    dataframe = filter_rare_relevancies(dataframe)
    for col_name in ['sem_sim_product_title', 'sem_sim_product_description']:
        filter_low_similarities(dataframe, col_name)

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

def show_feature_importances(dataframe: pd.DataFrame) -> None:
    """Calls a function that plots the importance of each feature that was used in the regression model"""
    
    dataframe = dataframe.copy()
    dataframe = filter_rare_relevancies(dataframe)
    for col_name in ['sem_sim_product_title', 'sem_sim_product_description']:
        filter_low_similarities(dataframe, col_name)

    y = dataframe['relevance'].values
    relevant_columns = ['sim_sim_product_title', 'sim_sim_product_description',
                        'sem_sim_product_title', 'sem_sim_product_description',
                        'len_of_query']
    X = dataframe[relevant_columns].values

    X_train, _, y_train, _ = TTS(X, y, test_size=0.2, random_state=40)

    RFR = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    
    RFR.fit(X_train, y_train)

    plot_feature_importances(relevant_columns, RFR.feature_importances_)

