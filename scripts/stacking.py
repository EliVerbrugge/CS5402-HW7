# Stacking in Python
import numpy
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
import joblib
import pickle

from os import path
import warnings
warnings.filterwarnings('ignore')

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


# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    #Define evaluation procedure
    cv=StratifiedKFold(n_splits=5)
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

X, y = get_dataset()
X = X.to_numpy()
y = y.to_numpy()


# KNeighborsClassifier is nearest neighbor (like kd trees),
# GaussianNB is simple Bayesian network,
# LogisticRegression is kind of like linear regression
models = {
    'KNN': KNeighborsClassifier(n_neighbors=1),
    'Random Forest': RandomForestClassifier(random_state=1),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf'),
    'GBM': joblib.load(path.join('..', 'models', 'gbm_cv10_n100_606acc.joblib'))
}


print("\nSummary of individual classifier\n")


lr = LogisticRegression()
# Create the ensemble classifier
sclf = StackingClassifier(classifiers=list(models.values()), meta_classifier=lr)
# Report the accuracy of the base classifiers
# zip makes tuples from item in 1st list and item in 2nd list
results = []
for label, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(),label))


print("\nSummary of stacking classifier\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sclf.fit(X_train, y_train)


summary(sclf)

acc = metrics.accuracy_score(y_test, sclf.predict(X_test))
acc_val = int(acc*1000)

print(acc)

joblib.dump(model, path.join('..', 'models', f'stacking_cv5_{acc_val}acc.joblib'))

train_test = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test
}

with open(path.join('..', 'models', f'stacking_train_test_{acc_val}.pkl'), 'wb') as outfile:
        pickle.dump(train_test, outfile)
