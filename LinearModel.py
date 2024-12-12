# Linear regression model for predicting MasterCard stock prices
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# Mastercard dataset obtained from: https://www.kaggle.com/datasets/kalilurrahman/mastercard-stock-data-latest-and-updated/data
# Some portions of the code may have been reused from previous assignments

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import matplotlib.pyplot as plt

class MyLinearRegression:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.X_temp = None
        self.y_temp = None
        self.r2_val = None
        self.y_pred = None
        self.mse_val = None

    def read_csv(self):
        # import data 
        self.data = pd.read_csv('Mastercard_stock_history.csv', sep=',', header=0, encoding = 'utf-8')

        # Data pre-processing
        # Must make the string datatype to datetime object, then sort by date
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.sort_values(by='Date', inplace=True)

        # Separate needed features from non-relavant ones, price prediction means closing price is the target variable
        # Dividends, stock splits are not needed
        features = self.data[['Open', 'High', 'Low', 'Volume']]
        target = self.data['Close']
        # Split data into 70% training, 15% validation, 15% test data
        # keep the random state as 42 in both splits for reproducibility.
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(features, target, test_size=0.3, random_state=42, shuffle=False)
        # With the remaining 30% of the data, split in half so we get 15% validation and 15% test
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42, shuffle=False)


    def model_fit(self):
        # Fit LinearRegression model here
        # Uses imported Linear Regression class and fits the training features/target label
        # X_train shape: (2710, 4), y_train shape: (2710,)
        self.model = LinearRegression().fit(self.X_train, self.y_train)
        # prediction, r2_score, and mse for validation data
        y_pred_val = self.model.predict(self.X_val)
        self.r2_val = r2_score(self.y_val, y_pred_val)
        self.mse_val = mean_squared_error(self.y_val, y_pred_val)
    

    def model_predict(self):
        assert self.model is not None, "Initialize the model, i.e. fill in and run the model_fit function"
        # Predictions made using the predict method
        y_pred_train = self.model.predict(self.X_train)
        self.y_pred = self.model.predict(self.X_test)
        # Difference between the actual price and the predicted price
        # can be seen by uncommenting the print statement below
        df = pd.DataFrame({'Predicted Price ($)':self.y_pred, 'Actual Price ($)': self.y_test,})
        #print(df)
        # Accuracy is reserved for classification problems, in linear regression, r2_score, mse is used
        # r2_score and mse for the training data
        r2_train = r2_score(self.y_train, y_pred_train)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        # root_mean_squared_error requires sklearn version 1.4+
        # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_error.html
        # Upgrade sklearn with command in terminal: python -m pip install --upgrade --user scikit-learn
        #root_mse = root_mean_squared_error(self.y_test, self.y_pred)
        # Calculations for test r2_score and MSE used instead of rMSE, print all r2_scores and MSEs to terminal
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Training R2-score: {r2_train}, Training MSE: {mse_train}")
        print(f"Validation R2-score: {self.r2_val}, Validation MSE: {self.mse_val}")
        print(f"Test R2-score: {r2}, Test MSE: {mse}")

        with open("r2_train_file.txt", "w") as file:
            file.write(str(r2_train) + ",")

        with open("r2_val_file.txt", "w") as file:
            file.write(str(self.r2_val) + ",")

        with open("r2_file.txt", "w") as file:
            file.write(str(r2) + ",")

    
    def visual(self):
        # Plotting the actual price vs predicted price
        # The test data and the dates have to be matched
        dates = self.data['Date'].iloc[-len(self.y_test):]
        plt.figure(figsize=(12, 6))
        plt.plot(dates, self.y_test, label="Actual Price", color="blue", linewidth=2)
        plt.plot(dates, self.y_pred, label="Predicted Price", color="orange", linewidth=2)
        plt.title(" Actual vs Predicted Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    classifier = MyLinearRegression()
    classifier.read_csv()
    classifier.model_fit()
    classifier.model_predict()
    classifier.visual()