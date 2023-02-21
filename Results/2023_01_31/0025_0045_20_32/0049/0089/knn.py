import pandas as pd
import glob
import math
from Utils import *

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Target Variables of Interest: biomass, lai (Leaf Area Index), yield
target_var_list = ["biomass", "lai", "yield"]
target_var = target_var_list[2]

stop_at_first = True

print("K-Nearest Neighbors Machine Learning\n-------------------------------")

for ens in ["Generic", "Generic2", "Generic3", "Generic4"]:
    # Read in all the daily output files and combine them into a single dataframe
    all_files = glob.glob("*.csv")
    df_list = []
    for filename in all_files:
        output_df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(output_df)
    output_df = pd.concat(df_list, axis=0, ignore_index=True)
    met_df = read_met("met00000.met")
    df = merge_output_met(met_df, output_df)

    # Filter rows by selected ensemble
    df = df[df['ensemble'] == ens]

    # Remove ensemble column
    df.drop(columns=['ensemble'], inplace=True)

    # Drop sim column
    df.drop(columns=['sim'], inplace=True)

    # Split the dataframe into features and target
    X = df.drop(columns=target_var_list)  # features are all columns except biomass, lai, and yield
    y = df[target_var]  # target variable is stored in target_var

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Implement K-Nearest Neighbors Regression
    knn = KNeighborsRegressor(n_neighbors=5)

    # Fit the model to the training data
    knn.fit(X_train, y_train)

    # Use the model to make predictions on the testing data
    y_pred = knn.predict(X_test)

    print("\n-----Results for ensemble = " + ens + "-----")

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error:", mse)

    # Calculate the R^2 score of the predictions
    r2 = knn.score(X_test, y_test)
    print("R^2 score:", r2)

    # Calculate R, the correlation coefficient
    corr_coef = math.sqrt(r2)
    print("r value:", corr_coef)

    if stop_at_first:
        break
