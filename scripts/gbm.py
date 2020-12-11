
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
import pandas as pd
from os import path

DATA_PATH = path.join('..', 'data', 'credit_output_binned.csv')

# Make a dataset
def get_dataset():
    df = pd.read_csv(DATA_PATH)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # X,y=make_classification(
    #     n_samples=1000,
    #     n_features=20,
    #     n_informative=15,
    #     n_redundant=5,
    #     random_state=7
    # )
    return X, y

# Get a list of models to evaluate
def get_models():
    models = dict()
    # Define number of trees to consider
    n_trees=[10,50,100,500]
    for n in n_trees:
        models[str(n)]=GradientBoostingClassifier(n_estimators=n)
    return models

# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    #Define evaluation procedure
    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
    # Evaluate the model and collect the results
    scores=cross_val_score(model,X,y,scoring='accuracy',cv=cv,n_jobs=-1)
    return scores

X, y = get_dataset()  # Define dataset
models = get_models() # Get the models to evaluate
results, names = list(), list() # Evalthe models, store results
for name, model in models.items():
    scores = evaluate_model(model, X, y) #Evaluate the model
    results.append(scores) # Store the results
    names.append(name) # Summarize the performance along the way
    print('>%s %.3f (%.3f)'%(name,mean(scores),std(scores)))

# Plot model performance for comparison
pyplot.boxplot(results,labels=names,showmeans=True)
pyplot.show()