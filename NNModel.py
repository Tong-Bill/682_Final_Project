# LSTM Neural Network model for predicting MasterCard stock prices
# https://scikit-learn.org/1.5/modules/neural_networks_supervised.html
# Mastercard dataset obtained from: https://www.kaggle.com/datasets/kalilurrahman/mastercard-stock-data-latest-and-updated/data
# Some portions of the code may have been reused from previous assignments

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
# Old importing methods for keras layers and optimizers do not work anymore
# New versions of tensorflow and keras have been moved to api subdirectory
from keras.api.layers import *
from keras.api.optimizers import *
from keras.src.callbacks import EarlyStopping, Callback
from keras import layers
import matplotlib.pyplot as plt

class MyNeuralNetwork:
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
        self.scaler = None
        self.X_norm = None
        self.Y_norm = None
        self.r2_val = None
        self.mse_val = None
        self.y_pred = None
        self.epochs = 0
        self.length = 0
        self.training_loss = []
        self.validation_loss = []
        self.training_r2 = []
        self.validation_r2 = []

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
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X_norm, self.Y_norm, test_size=0.3, random_state=42, shuffle=False)
        # With the remaining 30% of the data, split in half so we get 15% validation and 15% test
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42, shuffle=False)
        
        # Create sequences for LSTM, solves 'IndexError: tuple index out of range'
        # Calls the function below to actually make sequence ready for input into the NN
        self.length = 10  
        self.X_train, self.y_train = self.make_seq(self.X_train, self.y_train, self.length)
        self.X_val, self.y_val = self.make_seq(self.X_val, self.y_val, self.length)
        self.X_test, self.y_test = self.make_seq(self.X_test, self.y_test, self.length)
    
    # Takes parameters of features as X_values, and targets as y_values, along with the set sequence length
    # Returns np array of the results
    def make_seq(self, X_values, y_values, length):
        X_list = []
        y_list = []
        for i in range(len(X_values) - length):
            X_list.append(X_values[i:i + length])
            y_list.append(y_values[i + length])
        return np.array(X_list), np.array(y_list)
    
    # The LSTM model was not introduced in the AI course, below were tutorials/resources on it
    # https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/
    # https://keras.io/api/layers/recurrent_layers/lstm/
    # Taken and modified from the sample code provided by Github user 'omerbsezer':
    # https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/blob/master/StockPricesPredictionProject/pricePredictionLSTM.py
    def build_model(self):
        self.model = Sequential([
            # Units AKA neurons in the layer learn to capture patterns in the data, 50 & 100 are common
            # return_sequence determines whether the LSTM layer returns full sequence of outputs or just output from final time step
            # First LSTM Layer
            LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            # Observations from experimenting with droupout rate:
            # - 30% is too much being dropped, while 20% is slightly better than 10%
            # - 0.2 droupout or 20% seems to be a common value that doesn't lead to underfit
            # First regularization layer
            layers.Dropout(0.2),
            
            # Uncomment below for additional LSTM layer and Dropout layer, not needed
            #LSTM(units=50, return_sequences=True),
            #layers.Dropout(0.2),
            
            # Second LSTM Layer
            LSTM(units=50),
            # Second regularization layer
            layers.Dropout(0.2),
            # Output layer
            Dense(units=1),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mse'])
    '''
    Experimenting with dropout rate
    ---------------------------------
    30% Dropout results:
    Training R2-score: 0.9924867653617088, Training MSE: 4.6551704708389654e-05
    Validation R2-score: 0.9795696856225533, Validation MSE: 0.00022702584605700072
    Test R2-score: 0.7331060371239817, Test MSE: 0.0028014364194803012

    20% Dropout results:
    Training R2-score: 0.9975243317557556, Training MSE: 1.533914254116973e-05
    Validation R2-score: 0.9852608036240289, Validation MSE: 0.0001637849748973519
    Test R2-score: 0.9017836916431083, Test MSE: 0.0010309215699484335

    10% Dropout results:
    Training R2-score: 0.9917769211307755, Training MSE: 5.094987149249965e-05
    Validation R2-score: 0.9833231405733365, Validation MSE: 0.00018531668436250158
    Test R2-score: 0.8846268991382288, Test MSE: 0.0012110068099692557
    '''


    def model_fit(self):
        # Number of epochs and batch_size set here
        self.epochs = 50
        batch_size = 24
        # Early stopping basically stops training when the metric being monitored doesn't improve anymore
        # Validation loss is used as a monitor
        # patience is the threshold number of epochs with no improvement before being stopped
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # A custom logger is used to log r2_score for each epoch, since fit() does not keep track of it
        record_r2 = CustomLogger(self.X_train, self.y_train, self.X_val, self.y_val)
        h = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=batch_size, validation_data=(self.X_val, self.y_val), callbacks=[early_stopping, record_r2], shuffle=True)
        # Comment out above and uncomment below to disable early stopping
        #h = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_val, self.y_val), callbacks=[record_r2], shuffle=True)
        # prediction on validation set and calculate validation r2 score
        y_pred_val = self.model.predict(self.X_val)
        self.r2_val = r2_score(self.y_val, y_pred_val)
        self.mse_val = mean_squared_error(self.y_val, y_pred_val)
        # Set training, validation loss and r2_scores to initialized variables for plotting
        self.training_loss = h.history['loss']
        self.validation_loss = h.history['val_loss']
        self.training_r2 = record_r2.r2_training_list
        self.validation_r2 = record_r2.r2_validation_list

    def model_predict(self):
        print("\n")
        # Predictions made using the predict method for training and test data
        y_pred_train = self.model.predict(self.X_train)
        self.y_pred = self.model.predict(self.X_test)
        # r2_score and mse for training data
        r2_train = r2_score(self.y_train, y_pred_train)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
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
        # Reverse the scaler operation done at the start that normalized the inputs 
        y_pred_initial = self.scaler.inverse_transform(self.y_pred)
        y_test_initial = self.scaler.inverse_transform(self.y_test)

        # Plotting the actual price vs predicted price
        # The test data and the dates have to be matched
        dates = self.data['Date'].iloc[-len(self.y_test):]
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_test_initial, label="Actual Price", color="blue", linewidth=2)
        plt.plot(dates, y_pred_initial, label="Predicted Price", color="orange", linewidth=2)
        plt.title(" Actual vs Predicted Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def visual_2(self):
        # First graph details the training loss vs test loss
        epochs = range(1, len(self.training_loss) + 1)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.training_loss, label='Train Loss', color="yellow", linewidth=2)
        plt.plot(epochs, self.validation_loss, label='Test Loss', color="orange", linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        # Second graph details the training r2_score vs test r2_score
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.training_r2, label='Train R2', color="yellow", linewidth=2)
        plt.plot(epochs, self.validation_r2, label='Test R2', color="orange", linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('R2 Score')
        plt.title('Training and Test R2 Score over Epochs')
        plt.legend()

        plt.grid(True)
        plt.show()

# https://www.tensorflow.org/guide/keras/writing_your_own_callbacks
# on_epoch_end() is a special method inherited from from keras.src.callbacks
# Note: Assumed from tensorflow tutorial that was a random name, which is NOT the case
class CustomLogger(Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.r2_training_list = []
        self.r2_validation_list = []

    def on_epoch_end(self, epoch, logs=None):
        # Prediction and r2_score for training data
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        r2_training_list = r2_score(self.y_train, y_train_pred)
        self.r2_training_list.append(r2_training_list)
        # Prediction and r2_score for validation data
        y_val_pred = self.model.predict(self.X_val, verbose=0)
        r2_validation_list = r2_score(self.y_val, y_val_pred)
        self.r2_validation_list.append(r2_validation_list)

if __name__ == '__main__':
    classifier = MyNeuralNetwork()
    classifier.read_csv()
    classifier.build_model()
    classifier.model_fit()
    classifier.model_predict()
    classifier.visual()
    classifier.visual_2()