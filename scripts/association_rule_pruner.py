import operator
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


OPERATOR_MAP = {
    0: operator.lt,
    1: operator.le,
    2: operator.gt,
    3: operator.ge,
    4: operator.eq,
    5: operator.ne    
}


def _get_rule_index_pair(df, encoded_rule):
    """
    Returns a tuple of pd.Index, (idx, not_idx) such that 
    For all in idx the rule is true in df 
    and for all in not_idx the rule is false in df

    df              : Dataframe to be used
    encoded_rule    : Encoded rule, [column_idx, op_code, value]
    """
    col_idx, op_code, v = encoded_rule
    col = df.columns[col_idx]
    op = OPERATOR_MAP[op_code]

    rule = pd.Index(np.where(op(df[col], v))[0])
    not_rule = df.index.difference(rule)
    return (rule, not_rule)


def _get_rule_index_pairs(df, encoded_rules):
    """
    Returns list of tuples of index pairs, as specified by 
    """
    return [_get_rule_index_pair(df, encoded_rule) for encoded_rule in encoded_rules]


def _intersect_idxs(idxs):
    """
    Returns the intersection of an in indexable iterable of indexes
    """
    if len(idxs) == 1:
        return idxs[0]
    elif len(idxs) == 0:
        return pd.Index([])
    else:
        return idxs[0].intersection(_intersect_idxs(idxs[1:]))


def error_rate(errors, trials, significance):
    """
    Returns the expected error for a given amount of errors and trials for a specified signifigance
    errors [0, trials]
    trials [1, inf)
    significance (0, 1)
    """
    ci_low, _ = proportion_confint(trials-errors, trials, significance, method='beta')
    return (1 - ci_low)


def _calc_yn_en(ante_pairs, cnsq_pairs, n=None):
    """
    Calculates y_n and e_n given a pairs index & not_indexes 
    for antecedents and consequents

    When n is given, the n'th term in the antecedent is negated
    If n is not given, no term in the antecedent is negated.

    Returns (y, e)
    """
    # Generate indexes for c and !c
    cnsq = _intersect_idxs([c[0] for c in cnsq_pairs])
    not_cnsq = _intersect_idxs([c[1] for c in cnsq_pairs])

    # Determine indexes where A1, A2, .., !An, .., Am-1, Am is true
    ante = [a[0] for a in ante_pairs]
    if n is not None:
        ante[n] = ante_pairs[n][1]
    idxs = _intersect_idxs(ante)

    # Find where c (y), and !c (e)
    y = cnsq.intersection(idxs).size
    e = not_cnsq.intersection(idxs).size
    return (y, e)


def _calc_rule_error(ys, es, confidence):
    """
    Calcuates the expected error of a rule given a list 
    of y's and e's and a confidence level
    """
    errors = sum(es)
    trials = sum(ys) + errors
    return error_rate(errors, trials, confidence)



def _drop_condition(ante_pairs, cnsq_pairs, confidence):
    """
    Returns the index of the condition in the ante
    """
    if len(ante_pairs) == 0:
        # No rules to drop, return None
        return None

    # Calculate predicted error without modification of rule
    y, e = _calc_yn_en(ante_pairs, cnsq_pairs)
    if (y + e) == 0:
        # If no instances in dataset, don't modify rule
        # Return None as no condition should be dropped
        return None

    base_error = _calc_rule_error([y], [e], confidence)
    rule_errors = []

    # Calculate error for each rule
    for n in range(len(ante_pairs)):
        yn, en = _calc_yn_en(ante_pairs, cnsq_pairs, n=n)
        rule_errors.append(_calc_rule_error([y, yn], [e, en], confidence))

    if min(rule_errors) < base_error:
        return rule_errors.index(min(rule_errors))
    else:
        return None



def _simplify_rule(df, rule, confidence):
    """
    Returns a simplified rule with <= predicted error rate using 
    confidence value specified
    """
    ante, cnsq = rule
    ante_pairs = _get_rule_index_pairs(df, ante)    
    cnsq_pairs = _get_rule_index_pairs(df, cnsq)    

    # Walrus operator. Requires Python 3.8+
    while (cond_to_drop := _drop_condition(ante_pairs, cnsq_pairs, confidence)):
        # If condition to drop is not None, pop condition
        ante.pop(cond_to_drop)
        ante_pairs.pop(cond_to_drop)
    return [ante, cnsq]



def simplify_rules(df, rules, confidence):
    """
    Simplifies a set to rules based on a given confidence interval

    df      : Dataframe that the 
    """
    new_rules = []

    for idx, rule in enumerate(rules):
        simplified_rule = _simplify_rule(df, rule, confidence)
        if simplified_rule not in new_rules:
            # Don't add duplicate rules
            new_rules.append(simplified_rule)

    # Sort rules and return
    return sorted(new_rules, key=lambda x: str(x[0]))



if __name__ == '__main__':
    from copy import deepcopy
    CSV_PATH = r'..\data\credit_output_final_binned.csv'

    df = pd.read_csv(CSV_PATH)

    rules = [[[[14, 4, 'none']], [[6, 4, 0]]],
[[[18, 4, 1]], [[6, 4, 0]]],
[[[10, 4, 'none']], [[6, 4, 0]]],
[[[10, 4, 'none'],[18, 4, 1]], [[6, 4, 0]]],
[[[14, 4, 'none']], [[18, 4, 1]]],
[[[10, 4, 'none']], [[18, 4, 1]]],
[[[6, 4, 0],[10, 4, 'none']], [[18, 4, 1]]],
[[[6, 4, 0]], [[18, 4, 1]]],
[[[18, 4, 1]], [[10, 4, 'none']]],
[[[6, 4, 0],[18, 4, 1]], [[10, 4, 'none']]]
]

    print(f'Original ruleset has {len(rules)} rules.')
    print()
    # Generate simplified rules
    print('Creating rules for 95% confidence')
    rules_95_conf = simplify_rules(df, deepcopy(rules), .95)
    print(f'95% confidence ruleset has {len(rules_95_conf)} rules.')
    print()
    print('Creating rules for 25% confidence')
    rules_25_conf = simplify_rules(df, deepcopy(rules), .25)
    print(f'25% confidence ruleset has {len(rules_25_conf)} rules.')
    print()
    print("Writing rulesets to file...")


    # Write simplified rules to file
    with open(r'rules_95_conf.txt', 'w') as outfile:
        outfile.writelines([str(rule) + '\n' for rule in rules_95_conf])

    with open(r'rules_25_conf.txt', 'w') as outfile:
        outfile.writelines([str(rule) + '\n' for rule in rules_25_conf])

    print("Done!")

    