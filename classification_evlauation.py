import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the logistic regression model using the test set.

    Parameters:
        model (LogisticRegression): Trained logistic regression model
        X_test (pd.DataFrame): Test feature data
        y_test (pd.Series): Test target data
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Child', 'Parent']))

    #print("Confusion Matrix:")
    #print(confusion_matrix(y_test, y_pred)) 




    

