# Dakota Kosiorek

import os
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("data/data.csv")
    features = data.columns
    
    print(f"Null data summary:\n--------------------\n{data.isnull().sum()}\n\n")
    
    # Convert dataframe columns data type into int
    for f in features:
        data[f] = pd.to_numeric(data[f])
    
    # Split data
    y = data['y']
    X = data.drop(['y'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1776)

    # Classification models to train and test
    model_pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression()),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rfc': make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gbc': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        'gnb': make_pipeline(StandardScaler(), GaussianNB()),
        'dtc': make_pipeline(StandardScaler(), DecisionTreeClassifier()),
        'xgbc': make_pipeline(StandardScaler(), XGBClassifier()),
        'svc': make_pipeline(StandardScaler(), SVC())
    }
    
    # Train models
    print("Training models...")
    fit_models = {}
    for algorithm, pipeline in model_pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algorithm] = model
    print("Model training complete!\n")
    
    # Get accuracy of each model
    print("Getting model accuracy...")
    accuracy_models = {}
    for algorithm, model in fit_models.items():
        yhat = model.predict(X_test)
        accuracy_models[algorithm] = accuracy_score(y_test, yhat)
    print("Gathered all model accuracy!\n")
    
    for algorithm, accuracy in accuracy_models.items():
        accuracy_models[algorithm] = round(accuracy * 100, 2)
    
    plt.bar(accuracy_models.keys(), accuracy_models.values())
    plt.ylim((0, 100))
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Model Accuracy")
    plt.show()
    
    best_model = max(accuracy_models)
    print(model_pipelines[best_model].predict(X_test))
    best_model_accuracy = accuracy_models[best_model]
    
    predictions = pd.DataFrame();
        
    predictions['y'] = y_test
    predictions['y-hat'] = model_pipelines[best_model].predict(X_test)
    
    print(f"Predictions using {best_model} (Acc: {best_model_accuracy}%):\n{predictions}")
    
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.pie(predictions['y-hat'].value_counts(), 
            labels=predictions['y-hat'].unique(), 
            autopct=fmt_pie_values(predictions['y-hat'].value_counts())
    )
    ax1.legend()
    ax1.set_title(f"y-hat (uisng {best_model})")
    ax2.pie(predictions['y'].value_counts(), 
            labels=predictions['y'].unique(), 
            autopct=fmt_pie_values(predictions['y'].value_counts())
    )
    ax2.legend()
    ax2.set_title("y")
    fig.tight_layout()
    plt.show()

# For the pie chart: Gets the percent out of the pie chart slice and how many actual
#   values are in that slice
def fmt_pie_values(values):
    def my_fmt(x):
        total = sum(values)
        val = int(round(x*total/100.0))
        return '{:.1f}%\n({v:d})'.format(x, v=val)
    return my_fmt
 
if __name__ == "__main__":
    print("Starting program...\n")
    main()
    print("\n\nProgram complete!")