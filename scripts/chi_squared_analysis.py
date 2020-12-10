# Imports
from scipy.stats import chi2_contingency
from scipy.stats import chi2

from itertools import combinations

import pandas as pd


def is_dependent(df, attr1, attr2, significance=0.05):
    # Returns True if attr1 and attr2 in a specificied
    # dataframe are considered dependent using the Chi^2 test
    
    observation = create_observation_table(df, attr1, attr2)
    chi, pval, dof, exp = chi2_contingency(observation)
    
    p = 1 - significance
    
    critical_value = chi2.ppf(p, dof)
    
    return (chi > critical_value)


def create_observation_table(df, attr1, attr2):
    # Creates the observation table for two attributes
    # in a specified dataframe
    
    # Get unique values for attributes
    index = df[attr1].unique()
    cols = df[attr2].unique()

    # Sort elements in cols/index
    [arr.sort() for arr in [index, cols]]
    
    # Create empty table
    observation = pd.DataFrame([], index=index, columns=cols)
    
    # Insert data
    for idx, val in df.groupby([attr1, attr2]).size().items():
        row, col = idx
        observation[col].loc[row] = val
        
    observation.fillna(0, inplace=True)
        
    return observation


if __name__ == '__main__':
    # Read from data source
    DATA_SOURCE = r'../data/census_sanitized.csv'
    df = pd.read_csv(DATA_SOURCE)

    # List of all nominal attributes
    nominal_attributes = ['age',
                          'workclass',
                          'education',
                          'marital-status',
                          'occupation',
                          'relationship',
                          'race',
                          'sex',
                          'hours-per-week',
                          'native-country' ]

    
    # Iterate through combinations, determine dependence
    for c in combinations(nominal_attributes, 2):
        print(f'{str(c[0]) + " & " + str(c[1]):<35}: {is_dependent(df, *c)}')
    