Project Title: Superstore Customer Response Prediction and Analysis

Introduction:
This project aims to predict the likelihood of a customer giving a positive response to a marketing campaign. It serves as a practical demonstration of a full machine learning workflow, from exploratory data analysis to model building and the crucial step of handling an imbalanced dataset.

Problem Statement:
The primary goal is to build a predictive model that can identify customers likely to respond positively to a campaign. A secondary objective is to understand the factors that influence a customer's response. The dataset presents a challenge due to a significant class imbalance, which is addressed in the modeling process.

Methodology:

Data Preprocessing:

Initial data exploration to check for missing values and understand the data structure.

Feature selection based on domain knowledge (NumDealsPurchases, NumCatalogPurchases, NumStorePurchases, NumWebPurchases, NumWebVisitsMonth).

Numerical features are standardized using StandardScaler.

Principal Component Analysis (PCA) is applied to reduce dimensionality and visualize the data in 3D.

Model Building:

A deep neural network is built with TensorFlow/Keras to perform binary classification.

The model is initially trained on the raw data, revealing the challenges posed by the imbalanced nature of the dataset (a majority of "No" responses).

Handling Imbalanced Data:

The notebook demonstrates an attempt to correct the imbalance by calculating and applying class weights.

The results of this attempt are analyzed, showing that this particular approach did not improve model performance, providing an important insight into the complexity of such problems.

Key Findings:

The dataset is highly imbalanced, with a disproportionately low number of positive customer responses.

Initial model training without addressing the imbalance leads to a model that performs poorly on the minority class.

While class weights were tested as a potential solution, they did not lead to an improved model, suggesting the need for further exploration of more advanced techniques for imbalanced datasets (e.g., SMOTE, undersampling, etc.).

Technologies Used:

Python

Pandas

Numpy

Scikit-learn

TensorFlow/Keras

Matplotlib