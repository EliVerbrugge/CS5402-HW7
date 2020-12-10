import pandas as pd
from math import log

def partition(it):
    """
    Recursively produces all possible partitions of some finite iterable object
    """
    if len(it) == 1:
        # Base case
        # If only 1 element, then return the iterable
        yield [it]
        return

    for subpartition in partition(it[1:]):
        # Recursive case
        # Yield partitions by either
        #   1. Separating the first element into its own partition, or
        #   2. Placing the first element into every possible subpartion
        yield [[it[0]], *subpartition]    # Option 1
        for n, subset in enumerate(subpartition):
            yield [*subpartition[:n], [it[0], *subset], *subpartition[n+1:]]    # Option 2


def entropy(split):
    """
    Returns the entropy resulting from a given split
    """
    total = sum(split)
    f = lambda x: -x*log(x/total, 2)/total
    return sum([f(x) for x in split])


def weighted_entropy_from_splits(splits):
    """
    Returns the total entropy after a split
    """
    tups = [(sum(s), entropy(s)) for s in splits]
    total = sum([t[0] for t in tups])
    return sum([t[0]*t[1]/total for t in tups])


def gen_splits(df, split_attr, category_groups):
    """
    Returns splits between decision attributes given groups of attribute values to split on
    """
    decision_attr = df.columns[-1]
    # Creates different dfs such that each df only contains values in a specified category group
    grouped_dfs = [df[df[split_attr].isin(category_group)] for category_group in category_groups]
    splits = []
    for grouped_df in grouped_dfs:
        splits.append([len(grp[1]) for grp in grouped_df.groupby(decision_attr)])
    return splits
    

def determine_best_grouping(df, split_attr, verbose=False):
    """
    Returns a list groups of attributes that produces the greatest 
    information gain in a dataframe
    """
    # Create a list of every possible grouping of attribute values
    category_groupings = [*partition(list(df[split_attr].unique()))]

    # The last partition returned by partition() is one where all categories are combined
    # therefore no split it taken. For that reason that value is seperated from the rest
    category_groupings, no_split_group = category_groupings[:-1], category_groupings[-1]

    # Use the 'no_split_group' to determine entropy before split
    # (no_split_group) has all attributes grouped together, so there effectively is no split
    entropy_before_split = weighted_entropy_from_splits(gen_splits(df, split_attr, no_split_group))

    # Default value
    # One of the possible groupings should be a better grouping
    # so this will be replaced later
    best = (no_split_group, entropy_before_split)

    if verbose:
        print(f"Entropy before split: {round(entropy_before_split, 6)}")
        print()

    for cat_grp in category_groupings:
        # Iterate through each possible grouping
        # Determine split in decision attribute and calculate entropy
        splits = gen_splits(df, split_attr, cat_grp)
        entropy_after_split = weighted_entropy_from_splits(splits)

        if entropy_after_split <= best[1]:
            # Update best split if it provides a lower entropy
            best = (cat_grp, entropy_after_split)
        
        if verbose:
            information_gain = entropy_before_split - entropy_after_split
            print(f"Grouping: {str(cat_grp):40} -> Information gain = {round(information_gain, 6)}")

    if verbose:
        print('\n')
        print(f"Best grouping = {best[0]}")
        print(f"Entropy after splitting on grouping = {best[1]}")
        print(f"Information gain = {round(entropy_before_split - best[1], 6)}")

    # Return the grouping that did the best
    return best[0]


if __name__ == '__main__':
    FILEPATH = r'.\hw4_prob3.csv'
    df = pd.read_csv(FILEPATH)

    best_grouping = determine_best_grouping(df, df.columns[0], verbose=True)    


