# Logistic regression model for predicting MasterCard stock prices

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Mastercard_stock_history.csv")