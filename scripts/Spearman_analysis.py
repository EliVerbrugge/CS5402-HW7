# Imports
from scipy.stats import spearmanr
from itertools import combinations
from datetime import datetime

import pandas as pd


def is_dependent(df, attr1, attr2, threshold=0.8):
    # Uses spearman test to check if two attributes in the 
    # specified dataframe are dependent.
    X, Y = df[attr1], df[attr2]
    corr, pvalue = spearmanr(X, Y)
    
    # Attributes are likely dependent if >= threshold
    return abs(corr) >= threshold
    

if __name__ == '__main__':
    # Read from data source
    DATA_SOURCE = r'../data/census_sanitized.csv'
    df = pd.read_csv(DATA_SOURCE)

    # Read date as datetime object using MM/DD/YYYY format, convert to timestamp
    df['date-timestamp'] = df['date'].apply(
        lambda x: datetime.strptime(x, '%m/%d/%Y').timestamp()
    )

    # We change 'date' to 'date-timestamp' so that the date can be 
    # considered a continious number
    nonnominal_attributes = ['date-timestamp',
                             'population-wgt',
                             'education-num',
                             'capital-gain',
                             'capital-loss']

    # Iterate through combinations, determine dependence
    for c in combinations(nonnominal_attributes, 2):
        X, Y = df[c[0]], df[c[1]]
        print(f'{str(c[0]) + " & " + str(c[1]):<35}: {is_dependent(df, *c)}')
        