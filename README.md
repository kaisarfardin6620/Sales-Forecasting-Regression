# Sales Profit Prediction

## Project Overview

This project aims to build and evaluate various regression models to predict sales profit based on a given dataset. The process involves data cleaning, exploratory data analysis (EDA), feature engineering, model training and evaluation, hyperparameter tuning, and utilizing the PyCaret library for automated machine learning workflows.

## Dataset

The dataset used in this project is `sales.csv`, which contains information about sales transactions. The target variable for prediction is 'Profit'.

## Notebook Structure

The notebook is structured as follows:

1.  **Setup and Dependencies:** Installs necessary libraries like PyCaret.
2.  **Data Loading:** Loads the `sales.csv` dataset.
3.  **Data Exploration and Cleaning:**
    *   Displays dataset information (`info()`, `shape`, `describe()`, `head()`, `tail()`).
    *   Checks for missing values (`isnull().sum()`) and duplicates (`duplicated().sum()`).
    *   Performs basic data cleaning.
4.  **Feature Engineering:** Creates new features from the 'Date' column, such as 'Month', 'Quarter', 'DaysSince', 'Month_sin', 'Month_cos', 'DayOfWeek', and 'IsWeekend'. The original 'Date' column is then dropped.
5.  **Exploratory Data Analysis (EDA):**
    *   Generates histograms, pair plots, box plots, violin plots, and density plots to visualize the distribution and relationships of numerical features.
    *   Computes and visualizes the correlation matrix of numerical features using a heatmap.
    *   Identifies and visualizes outliers using box plots.
    *   Includes a function to remove outliers using the Interquartile Range (IQR) method and shows the shape of the DataFrame before and after outlier removal.
6.  **Data Preparation for Modeling:**
    *   Separates features (x) and target variable (y).
    *   Identifies categorical and numerical columns.
    *   Sets up a `ColumnTransformer` for one-hot encoding of categorical features and standard scaling of numerical features.
    *   Splits the data into training and testing sets.
7.  **Model Training and Evaluation (Manual):**
    *   Trains and evaluates several traditional regression models using scikit-learn pipelines:
        *   Linear Regression
        *   Decision Tree Regressor
        *   Random Forest Regressor
        *   Support Vector Regressor (SVR)
        *   K-Neighbors Regressor
    *   Defines an `evaluate_model` function to calculate and print key regression metrics (MSE, RMSE, MAE, R2).
    *   Visualizes the residuals for each trained model.
    *   Compares the performance metrics of all models using a bar plot.
8.  **Hyperparameter Tuning:**
    *   Defines a dictionary of models and their respective hyperparameter grids for tuning.
    *   Uses `GridSearchCV` to find the best hyperparameters for each model on the training data.
    *   Evaluates the performance of the tuned models on the test data and displays the best parameters and performance metrics.
    *   Generates actual vs. predicted plots for each tuned model.
9.  **Automated Modeling with PyCaret:**
    *   Sets up the PyCaret environment for regression by specifying the target variable and session ID for reproducibility.
    *   Compares various regression models automatically using `compare_models()`.
    *   Evaluates the best model identified by PyCaret using `evaluate_model()`.

## Requirements

*   Python 3
*   Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, pycaret

You can install the required libraries using pip:
Use code with caution
bash pip install pandas numpy matplotlib seaborn scikit-learn pycaret[full] pycaret[mlops] pycaret[time-series]

*(Note: The notebook includes these installation commands at the beginning)*

## How to Run

1.  Clone or download the notebook file.
2.  Upload the `sales.csv` dataset to your Google Drive.
3.  Run the notebook in Google Colab. Make sure to update the path to your dataset if necessary:
Use code with caution
python df = pd.read_csv('/content/drive/MyDrive/Dataset/sales.csv')

4.  Execute the cells sequentially to perform the data analysis, model training, and evaluation.

## Results

The notebook provides performance metrics for various regression models, both before and after hyperparameter tuning. It also utilizes PyCaret to automate the model comparison process and identify the best performing model. The visualizations help in understanding the data and the performance of the models.
