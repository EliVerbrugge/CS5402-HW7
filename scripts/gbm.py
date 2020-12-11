
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
import pandas as pd
from os import path
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import joblib
import pickle
from itertools import product


DATA_PATH = path.join('..', 'data', 'credit_output_binned.csv')

# Make a dataset
def get_dataset():
    df = pd.read_csv(DATA_PATH)
    for col in df:
        if df[col].dtype == 'O':
            # If type object (aka string), encode to numeric value
            encoder = LabelEncoder()
            df[col] = pd.Series(encoder.fit_transform(df[col]))
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y

# Get a list of models to evaluate
def get_models():
    models = dict()
    # Define number of trees to consider
    n_estimators=[10,50,100]
    max_features=[5,7,10]
    max_depth=[5,7,10]
    subsample=[0.2, 0.5, 0.8]
    settings_generator = product(n_estimators, max_features, max_depth, subsample)
    for settings in settings_generator:
        models[str(settings)]=GradientBoostingClassifier(
            n_estimators=settings[0],
            max_features=settings[1],
            max_depth=settings[2],
            subsample=settings[3]
        )
    return models

# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    #Define evaluation procedure
    cv=StratifiedKFold(n_splits=10)
    # Evaluate the model and collect the results
    scores=cross_val_score(model,X,y,scoring='accuracy',cv=cv,n_jobs=-1)
    return scores

X, y = get_dataset()  # Define dataset
models = get_models() # Get the models to evaluate
results, names = list(), list() # Eval the models, store results
for name, model in models.items():
    # Perform cross validation to determine optimal hyper-parameters of the classifier
    scores = evaluate_model(model, X, y) #Evaluate the model
    results.append(scores) # Store the results
    names.append(name) # Summarize the performance along the way
    print('>%s %.3f (%.3f)'%(name,mean(scores),std(scores)))

# Plot model performance for comparison
pyplot.boxplot(results,labels=names,showmeans=True)
pyplot.show()

# Cross validation has shown the the optimal hyper parameters is n_estimators=10
# However, after looking at confusion matrix for n=10, we see that it was just clasifying
# 'Good' for everyone. The next best option is n=50.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# model = models['50']
# model.fit(X_train, y_train) # Fit model with data now that hyper-parameters chosen via cross validation

# acc = metrics.accuracy_score(y_test, model.predict(X_test))
# print(acc)
# print(metrics.confusion_matrix(y_test, model.predict(X_test)))

# acc_val = int(acc*1000)

# joblib.dump(model, path.join('..', 'models', f'gbm_cv10_n50_{acc_val}acc.joblib'))

# train_test = {
#     'X_train': X_train,
#     'y_train': y_train,
#     'X_test': X_test,
#     'y_test': y_test
# }

# with open(path.join('..', 'models', f'gbm_train_test_{acc_val}.pkl'), 'wb') as outfile:
#         pickle.dump(train_test, outfile)
