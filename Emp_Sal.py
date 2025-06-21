# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:24:39 2025

@author: saidi
"""

# Om Sai Samartha

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\saidi\OneDrive\Desktop\SriKrishna\FSDS - AI Engineer\Class Notes & Assignments\Salary_Data.csv")

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Machine predicts the future
y_pred = regressor.predict(x_test)


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

# Optional: Output of the intercept and coefficient of linear model
print(f"Intercept: {regressor.intercept_}")
print(f"Coefficient: {regressor.coef_}")

y_12 = m_slope*12 + c_intercept
print(y_12)

# Comparison of actual and predicted salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test,y_test)
print(variance)

#Adding statistical code
data.mean()
data.std()
data['Salary'].mean()
data['Salary'].median()
data['Salary'].mode()
data.var()
data['Salary'].var()
data.std()
data['Salary'].std()

from scipy.stats import variation
variation(data.values)
variation(data['Salary'])
data.corr()

#This will give you the correlation between Salary and Years of Experience
data['Salary'].corr(data['YearsExperience'])

#This will give you the skewness of entire dataframe
data.skew()

data['Salary'].skew()

# This will give Standard error of entire dataframe

data.sem()

# This will give Standard error of particular column

data['Salary'].sem()

# This will give Z-score of entire dataframe
# Z-score range is -3 to 3

import scipy.stats as stats
data.apply(stats.zscore)

stats.zscore(data['Salary'])

a = data.shape[0] # this will give number of rows
b = data.shape[1] # this will give number of columns

degree_of_freedom = a - b
print(degree_of_freedom) # this will give degree of freedom of entire dataset

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(data.values)
SST = np.sum((data.values-mean_total)**2)
print(SST)

r_square = 1 - (SSR/SST)
r_square

import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())

