Project Title: Time-Series Sales Forecasting with XGBoost

Introduction:
This project predicts daily sales for a specific store and item using time-series analysis and machine learning. It demonstrates how to transform sequential data into a format suitable for a regression model by creating time-based features and leveraging a powerful gradient-boosting algorithm (XGBoost) for accurate forecasting.

Problem Statement:
The goal is to forecast future daily sales, which is a key business problem for inventory management and resource planning. The project focuses on a single time series, extracting crucial patterns and trends to make reliable predictions.

Methodology:

Data Preparation:

The raw sales data is loaded, filtered for a specific store and item, and then sorted and re-indexed to ensure a consistent daily frequency.

A custom function create_features() is used to engineer time-series features. This includes:

Lag features: Past sales values (e.g., from 1, 7, and 14 days prior) are used as predictors to capture immediate and weekly patterns.

Rolling mean features: Moving averages over specified windows (e.g., 7 and 30 days) are calculated to capture underlying trends.

Time-based features: The day of the week and month are extracted to account for seasonality and cyclical patterns.

Model Training:

The data is split into training and testing sets based on a specific date (2017-01-01).

An XGBoost Regressor model is trained on the engineered features. This model is chosen for its efficiency and strong performance on structured data.

Evaluation:

The model's predictions on the test set are evaluated using the Root Mean Squared Error (RMSE), a standard metric for regression tasks.

A plot is generated to visually compare the model's predictions against the actual sales data.

Feature importance is visualized to understand which engineered features contributed most to the model's predictive power. This helps in validating the feature engineering process and can guide future improvements.

Technologies Used:

Python

Pandas

Numpy

Scikit-learn

Matplotlib

XGBoost