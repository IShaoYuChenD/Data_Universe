Project Title: Credit Score Classification Model

Introduction:
This project develops and evaluates a machine learning model to classify credit scores into three categories: Good, Standard, and Poor. The solution utilizes a deep neural network built with TensorFlow/Keras and Scikit-learn for data preprocessing. The goal is to demonstrate a full end-to-end machine learning pipeline, from data preparation and feature engineering to model training and evaluation.

Problem Statement:
The objective is to predict a person's credit score based on various personal and financial attributes. This is a multi-class classification problem where the model must assign one of three labels: Good, Standard, or Poor.

Methodology:

Data Preprocessing:

Numerical features are handled using a SimpleImputer with a 'mean' strategy and StandardScaler to normalize the data.

Categorical features are imputed with the 'most frequent' value and then one-hot encoded using OneHotEncoder.

A custom transformer, CastToStrTransformer, was created to handle NaN values in categorical data before one-hot encoding.

A ColumnTransformer is used to apply different preprocessing steps to numerical and categorical features in a single step.

Model Architecture:

A sequential neural network with multiple dense layers and ReLU activation functions.

Dropout layers are included to prevent overfitting.

The final layer uses a softmax activation function for multi-class classification.

Training and Evaluation:

The model is compiled with the adam optimizer and categorical_crossentropy loss function.

The data is split into training and validation sets (80/20 split) to evaluate the model's performance on unseen data.

An EarlyStopping callback is used to monitor the validation loss and prevent overfitting, restoring the best weights.

The model's performance is visualized using plots for accuracy and loss over epochs.

A confusion matrix is generated to analyze the model's predictions and identify potential misclassifications.

Technologies Used:

Python

Pandas

Scikit-learn

TensorFlow/Keras

Matplotlib

Numpy