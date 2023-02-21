import pandas as pd
import glob
import math

from Utils import *

# Target Variables of Interest: biomass, lai (Leaf Area Index), yield
target_var_list = ["biomass", "lai", "yield"]
target_var = target_var_list[2]

stop_at_first = True

print("Neural Network Machine Learning\n-------------------------------")

for ens in ["Generic", "Generic2", "Generic3", "Generic4"]:
    # Read in all the daily output files and combine them into a single dataframe
    all_files = glob.glob("*.csv")
    df_list = []
    # print(all_files)
    for filename in all_files:
        output_df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(output_df)
    output_df = pd.concat(df_list, axis=0, ignore_index=True)
    met_df = read_met("met00000.met")
    df = merge_output_met(met_df, output_df)
    # print(df.columns)

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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Implement a neural network
    from tensorflow import keras

    # Define the architecture of the neural network
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Fit the model to the training data
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

    # Visualize the training history
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = model.predict(X_test)

    print("\n-----Results for ensemble = " + ens + "-----")

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error:", mse)

    # Calculate the R^2 score of the predictions
    r2 = r2_score(y_test, y_pred)
    print("R^2 score:", r2)

    # We can also calculate R, correlation coefficient
    corr_coef = math.sqrt(r2)
    print("r value:", corr_coef)

    if stop_at_first == True :
        break
