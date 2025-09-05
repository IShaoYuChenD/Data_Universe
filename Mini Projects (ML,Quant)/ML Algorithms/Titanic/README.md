Project Title: Titanic Survival Prediction with Logistic Regression

Introduction:
This project predicts the survival of passengers on the Titanic using a logistic regression model. It provides a clear, step-by-step example of a machine learning classification pipeline, from data preprocessing to comprehensive model evaluation.

Problem Statement:
The objective is to predict whether a passenger survived the Titanic disaster based on a set of features, including their class, age, gender, and ticket information. This is a fundamental binary classification problem used to demonstrate core machine learning concepts.

Methodology:

Data Preprocessing Pipeline:

A Scikit-learn Pipeline is used to automate the data cleaning and transformation steps.

Numerical features (Age, SibSp, Parch, Fare) are handled by imputing missing values with the median and then scaled using StandardScaler.

Categorical features (Pclass, Sex, Embarked) are handled by imputing missing values with the most frequent value and then converted to numerical format using OneHotEncoder.

The ColumnTransformer is used to apply these different preprocessing steps to the correct columns simultaneously.

Model Training:

A Logistic Regression model is trained on the preprocessed data. This model is a great choice for this problem due to its simplicity and interpretability.

Evaluation:

The model's performance is thoroughly evaluated using several metrics and visualizations:

Classification Report: Provides key metrics like precision, recall, and F1-score for each class (survived/not survived).

ROC AUC Score: A single metric that summarizes the model's ability to distinguish between the two classes. An AUC score of 0.86 on the training data indicates strong performance.

Confusion Matrix: A visualization that shows the number of correct and incorrect predictions for each class, providing a clear picture of true positives, false positives, true negatives, and false negatives.

ROC Curve: A plot of the True Positive Rate vs. the False Positive Rate, which visually represents the model's performance across different classification thresholds.

Technologies Used:

Python

Pandas

Scikit-learn

Matplotlib

Seaborn