# src/models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd

def train_multioutput_rf(X_train, y_train, n_estimators=200, max_depth=5, random_state=42):
    """
    Train a multi-output RandomForestClassifier.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    clf = MultiOutputClassifier(rf) # trains a separate Random Forest model for each stock
    clf.fit(X_train, y_train)
    return clf

def predict_rank_probabilities(clf, X_test, top_class=2):
    """
    Return predicted probabilities of the top quantile for ranking.
    """
    y_proba = [est.predict_proba(X_test) for est in clf.estimators_] 
    # Each estimator corresponds to one of the stocks we are predicting
    # probability for each class
    pred_ranks = pd.DataFrame(
        {ticker: proba[:, top_class] for ticker, proba in zip(range(len(y_proba)), y_proba)}, #only the column corresponding to the top class's probability
        index=X_test.index
    )
    return pred_ranks
