import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from jupyterthemes import jtplot

# Set the plotting style
jtplot.style(theme='monokai', ticks=True, grid=False)

# Read the data
mpg_df = pd.read_csv('FuelEconomy.csv')

# Prepare the data 
X = mpg_df['Horse Power'].values.reshape(-1, 1)
Y = mpg_df['Fuel Economy (MPG)'].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the Simple Linear Regression model
SimpleLinearRegression = LinearRegression(fit_intercept=True)
SimpleLinearRegression.fit(X_train, Y_train)

# Set the figure size and plot the train and test data 
plt.figure(figsize=(13, 7))
plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='yellow')

# Find how far the data ranges on the x-axis and plot the regression line using this range
X_combined = np.concatenate((X_train, X_test))
X_combined_sorted = np.sort(X_combined, axis=0)
plt.plot(X_combined_sorted, SimpleLinearRegression.predict(X_combined_sorted), color='red')

# Add the plot labels
plt.ylabel('Fuel Efficiency [MPG]')
plt.xlabel('Horsepower')
plt.title('Impact of Horsepower on Fuel Efficiency')

plt.show()

# Assess trained model performance
accuracy_LinearRegression = SimpleLinearRegression.score(X_test, Y_test)
print('Accuracy of Linear Regression Model: ', round(accuracy_LinearRegression * 100, 1), '%')

# Predict the fuel economy (MPG) for a car with 50 horsepower
horsepower_value = np.array([50])
horsepower_value = horsepower_value.reshape(-1, 1)
predicted_mpg = SimpleLinearRegression.predict(horsepower_value)
print('Predicted Fuel Economy (MPG):', round(predicted_mpg[0][0], 1))