PROJECT NAME: Impact of Horsepower on Fuel Efficiency using Simple Linear Regression

## OVERVIEW
This project is a simple linear regression model that predicts fuel efficiency (MPG) 
based on horsepower data, utilizing Python libraries like Pandas, Numpy, Seaborn, 
Matplotlib, and Scikit-learn.

## TABLE OF CONTENTS
1. Installation
2. Usage
3. Features
4. Documentation
5. Credits

### Prerequisites
- Python 3.10.9 (this is the version used for development and testing)
- Third-party libraries: `pandas`, `numpy`, `seaborn`, `matplotlib.pyplot`, 
`jupyterthemes`, `sklearn.model_selection`, `sklearn.linear_model`

### Installation Steps 
1. Clone the repository:
git clone https://github.com/ocampbell378/FuelEfficiencyPrediction.git
2. Install the required libraries:
pip install -r requirements.txt

## USAGE
To run the project, use the following command:
python main.py

## FEATURES
### Feature 1: Data Visualization of Horsepower vs. Fuel Efficiency
The script uses Seaborn and Matplotlib to create a scatter plot with a regression line, 
visualizing the relationship between horsepower and fuel efficiency (MPG) from a fuel 
economy dataset.

### Feature 2: Linear Regression Model for Predicting Fuel Efficiency
The script implements a Simple Linear Regression model using Scikit-learn to predict 
fuel efficiency based on horsepower. It also calculates the accuracy of the model and 
provides a predicted fuel economy for a given horsepower.

## DOCUMENTATION
### Modules and Functions

**main.py**: Handles data visualization, linear regression model creation, and prediction.

- `import pandas as pd, import numpy as np, import seaborn as sns, import matplotlib.pyplot as plt`: Imports necessary libraries for data manipulation, visualization, and plotting.
- `jtplot.style()`: Applies a custom theme to plots using Jupyter Themes.
- `pd.read_csv('FuelEconomy.csv')`: Loads the dataset from a CSV file into a DataFrame.
- `sns.regplot()`: Plots a regression line on the scatter plot of horsepower vs. fuel efficiency.
- `train_test_split()`: Splits the dataset into training and testing sets for model evaluation.
- `LinearRegression()`: Creates and fits a linear regression model to predict fuel efficiency based on horsepower.
- `score()`: Evaluates the accuracy of the linear regression model.
- `predict()`: Predicts fuel efficiency for a given horsepower using the trained model.

## CREDITS
Developed by Owen Campbell
The FuelEconomy.csv dataset is taken from the "Simple Linear Regression for the Absolute Beginner" course on Coursera.
