import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def read_csv_data(file_path):
    data = pd.read_csv(file_path)
    if not {'median', 'mad', 'type', 'person'}.issubset(data.columns):
        raise ValueError("CSV file must contain 'median', 'mad', 'type', and 'person' columns.")
    return data

def plot_data_with_regression(data, model):
    colors = {'parent': 'blue', 'child': 'green'}
    for person_type in data['type'].unique():
        subset = data[data['type'] == person_type]
        plt.scatter(subset['mad'], subset['median'], 
                   label=f"{person_type.title()}s", 
                   color=colors.get(person_type, 'gray'))

    mad_min, mad_max = data['mad'].min(), data['mad'].max()
    median_min, median_max = data['median'].min(), data['median'].max()
    buffer_mad = (mad_max - mad_min) * 0.1
    buffer_median = (median_max - median_min) * 0.1

    mad_range = np.linspace(mad_min - buffer_mad, mad_max + buffer_mad, 100)
    median_range = np.linspace(median_min - buffer_median, median_max + buffer_median, 100)
    grid = np.array([[mad, median] for mad in mad_range for median in median_range])
    
    probs = model.predict_proba(grid)[:, 1].reshape(len(mad_range), len(median_range))
    plt.contourf(mad_range, median_range, probs.T, levels=np.linspace(0, 1, 11), 
                cmap='RdYlBu', alpha=0.3)

    plt.xlabel('Median Absolute Deviation')
    plt.ylabel('Median')
    plt.title('Distribution of Medians and MADs with Non-Linear Decision Boundary')
    plt.colorbar(label='Probability of Parent')
    plt.legend(['Parents', 'Children'])
    plt.show()

def train_regression_model(data):
    X = data[['mad', 'median']]
    y = (data['type'] == 'parent').astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline with polynomial features, scaling, and regularized logistic regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=5, include_bias=False)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=0.1, max_iter=10000))
    ])
    
    model.fit(X_train, y_train)
    
    # Calculate and print accuracy on training and testing sets
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Testing Accuracy: {test_accuracy:.3f}")
    
    return model

if __name__ == "__main__":
    file_path = "C:/Users/lukev/Projects/BehavBuddy/csv_accurate/pitchAll_ROUNDED.csv"
    data = read_csv_data(file_path)
    model = train_regression_model(data)
    plot_data_with_regression(data, model)
    joblib.dump(model, 'C:/Users/lukev/Projects/BehavBuddy/trained_models/trainedBB_Afive.pkl')