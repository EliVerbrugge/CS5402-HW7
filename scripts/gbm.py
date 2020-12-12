
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
from pprint import pprint


DATA_PATH = path.join('..', 'data', 'credit_output_final_binned.csv')

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
    max_features=[5,7,10,20]
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

def run_alg(model, f):
    # This is normally run while in interpretted mode because we
    # needed to look at individual models. Might not run correctly
    # at regular runtime because of use of y_test and X_test
    return f(y_test, model.predict(X_test))

def summary(model):
    # This is normally run while in interpretted mode because we
    # needed to look at individual models. Might not run correctly
    # at regular runtime because of use of y_test and X_test
    print(f"Accuracy: {round(run_alg(model, metrics.accuracy_score), 4)}")
    print(f"Precision: {round(run_alg(model, metrics.precision_score), 4)}")
    print(f"Recall: {round(run_alg(model, metrics.recall_score), 4)}")
    print(f"ROC AUC: {round(run_alg(model, metrics.roc_auc_score), 4)}")
    print(f"F-measure: {round(run_alg(model, metrics.f1_score), 4)}")
    print(f"Kappa: {round(run_alg(model, metrics.cohen_kappa_score), 4)}")
    print("Confusion matrix")
    print(run_alg(model, metrics.confusion_matrix))

X, y = get_dataset()  # Define dataset
models = get_models() # Get the models to evaluate
results, names = list(), list() # Eval the models, store results
score_map = dict()
for name, model in models.items():
    # Perform cross validation to determine optimal hyper-parameters of the classifier
    scores = evaluate_model(model, X, y) #Evaluate the model
    results.append(scores) # Store the results
    names.append(name) # Summarize the performance along the way
    score_mean = mean(scores)
    score_map[name] = score_mean
    print('>%s %.3f (%.3f)'%(name,score_mean,std(scores)))



# Plot model performance for comparison
pyplot.boxplot(results,labels=names,showmeans=True)
pyplot.show()

sorted_models = sorted(score_map, key=lambda x: score_map[x], reverse=True)
# Cross validation has shown the the optimal hyper parameters is n_estimators=10
# However, after looking at confusion matrix for n=10, we see that it was just clasifying
# 'Good' for everyone.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# model = models['(50, 10, 10, 0.8)']   # unbinned
model = models['(100, 10, 10, 0.5)']    # binned, we found this one to be the best, ignoring n=10 as stated above
                                        # During cv, had a accuracy of 55.7%
model.fit(X_train, y_train) # Fit model with data now that hyper-parameters chosen via cross validation

summary(model)

acc = metrics.accuracy_score(y_test, model.predict(X_test))
acc_val = int(acc*1000)

joblib.dump(model, path.join('..', 'models', f'gbm_cv10_n100_{acc_val}acc.joblib'))

train_test = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test
}

with open(path.join('..', 'models', f'gbm_train_test_{acc_val}.pkl'), 'wb') as outfile:
        pickle.dump(train_test, outfile)
