# XGBoost model for predicting MasterCard stock prices
# https://xgboost.readthedocs.io/en/stable/python/python_intro.html#training
# Mastercard dataset obtained from: https://www.kaggle.com/datasets/kalilurrahman/mastercard-stock-data-latest-and-updated/data
# Some portions of the code may have been reused from previous assignments

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import matplotlib.pyplot as plt

class MyXGBoost:
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
        target = self.data['Close'].values.reshape(-1, 1)

        # To prepare data to neural network training, we need to normalize inputs in the range [0,1]
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.X_norm = self.scaler.fit_transform(features)
        self.Y_norm = self.scaler.fit_transform(target)
        # Split data into 70% training, 15% validation, 15% test data
        # keep the random state as 42 in both splits for reproducibility.
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(features, target, test_size=0.3, random_state=42, shuffle=False)
        # With the remaining 30% of the data, split in half so we get 15% validation and 15% test
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42, shuffle=False)

    # XGBoost was introduced as a concept in the course but was never implemented
    # Documentation for building an XGBoost model: https://xgboost.readthedocs.io/en/latest/get_started.html
    def model_fit(self):
        self.model = xgb.XGBRegressor()
        parameters = {
            'n_estimators': np.arange(1, 20),
            'objective': ['reg:squarederror', 'reg:squaredlogerror'],
            'booster': ['gblinear'],
            'base_score': [0.2, 0.5, 0.8]

            # Parameters for using Tree Booster, not used due to poor results compared with Linear Booster
            # Tree booster typically used for classification whereas linear booster for linear regression
            #'booster': ['gbtree'],
            #'learning_rate': [.0001, 0.01, 0.2, 0.3],
            #'max_depth': [3, 5, 7],
            #'subsample': [0.8, 1.0],
            #'colsample_bytree': [0.8, 1.0]
        }
        grid_model = GridSearchCV(estimator=self.model, param_grid=parameters, return_train_score=True)
        grid_model.fit(self.X_train, self.y_train)
        grid_model = grid_model.best_params_

        self.model = xgb.XGBRegressor(
            booster=grid_model['booster'], 
            n_estimators=grid_model['n_estimators'], 
            objective=grid_model['objective'],
            base_score=grid_model['base_score']

            # Parameters for Tree Booster 
            #learning_rate=grid_model['learning_rate'],
            #max_depth=grid_model['max_depth'],
            #tree_method='gpu_hist',
            #subsample=grid_model['subsample'],
            #colsample_bytree=grid_model['colsample_bytree']
        )      
        self.model.fit(self.X_train, self.y_train)


    def model_predict(self):
        # Predictions made using the predict method for training and test data
        y_pred_train = self.model.predict(self.X_train)
        self.y_pred = self.model.predict(self.X_test)
        # prediction on validation set and calculate validation r2 score
        y_pred_val = self.model.predict(self.X_val)
        self.r2_val = r2_score(self.y_val, y_pred_val)
        self.mse_val = mean_squared_error(self.y_val, y_pred_val)
        # r2_score and mse for training data
        r2_train = r2_score(self.y_train, y_pred_train)
        mse_train = r2_score(self.y_train, y_pred_train)
        # r2_score and mse for test data, output into terminal
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Training R2-score: {r2_train}, Training MSE: {mse_train}")
        print(f"Validation R2-score: {self.r2_val}, Validation MSE: {self.mse_val}")
        print(f"Test R2-score: {r2}, Test MSE: {mse}")

        with open("r2_train_file.txt", "a") as file:
            file.write(str(r2_train) + ",")

        with open("r2_val_file.txt", "a") as file:
            file.write(str(self.r2_val) + ",")

        with open("r2_file.txt", "a") as file:
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
    classifier = MyXGBoost()
    classifier.read_csv()
    classifier.model_fit()
    classifier.model_predict()
    classifier.visual()