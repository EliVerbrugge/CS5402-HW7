import pandas as pd
import numpy as np
import sys
import re
import argparse
import ast
from statsmodels.stats.proportion import proportion_confint

operator_to_int = {
 "<": 0,
 "<=": 1,
 ">": 2,
 ">=": 3,
 "=": 4,
 "==": 4,
 "!=": 5,
}
int_to_operator_opposite = {
 0: ">=",
 1: ">",
 2: "<=",
 3: "<",
 4: "!=",
 5: "==",
}
int_to_operator = {
 0: "<" ,
 1: "<=",
 2: ">",
 3: ">=",
 4: "==",
 5: "!=",
} 

def format_rule_entity(column_names, line):
 entity: list = []
 # Split on operators, but still keep the seperator
 if "<=" in line or ">=" in line or "==" in line:
     elements = re.split('(<=|>=|==|:)',line)
 else:
    elements = re.split('(<|<=|>|>=|=|==|!=|:)',line)

 print(elements)
 # Append the attribute index
 entity.append(column_names.index(elements[0].strip()))
 # Append the enumeration of the operator
 entity.append(operator_to_int[elements[1]])
 # Append the value itself
 entity.append(elements[2].strip('\n '))
 #print(entity)
 #print(line)
 return entity
 
def generate_rules(df, rulesFile):
 ruleLines = rulesFile.readlines()
 column_names: list = []
 for name in df.columns:
    column_names.append(name)

 # Create a list of parent antecendents for our row that we modify on the fly
 antecedents: list = []
 rules: list = []
 level: int = 0
 for line in ruleLines:
    num_vertical_bar = line.count('|')
    line = line.replace('|', '')
    if num_vertical_bar < level and len(antecedents) > 0:
        # We have moved back one level, remove an antecedent from our list
        diff = level - num_vertical_bar
        for i in range(diff):
            elem = antecedents.pop()
    # If we find a : we have found the end of a rule
    if ":" in line:
        #print(line)
        full_antecedents = antecedents.copy()

        elements = re.split(':',line) 

        full_antecedents.append(format_rule_entity(column_names, elements[0]))
        consequent = [len(column_names)-1, operator_to_int["="], elements[1].strip('\n ')]

        new_rule = [full_antecedents, consequent]
        rules.append(new_rule)
    else:
        # Add an entity to the list of antecdents as we haven't reached a leaf
        antecedents.append(format_rule_entity(column_names, line))

    level = num_vertical_bar

 f = open("rules_gen.txt", "w")
 for rule in rules:
    f.write(str(rule)+"\n")

def compare(df, column_name, operator, value):
 '''
 Returns a dataframe that has had the following condition: column_name
 operator (can be ==, !=, >, etc) value
 '''
 value_pos = re.sub("^-", "", value)
 if str.isdigit(value_pos):
     value = float(value)
 if operator == ">":
    return df.loc[df[column_name] > value]
 elif operator == ">=":
    return df.loc[df[column_name] >= value]
 elif operator == "<":
    print(value)
    return df.loc[df[column_name] < value]
 elif operator == "<=":
    return df.loc[df[column_name] <= value]
 elif operator == "==":
    return df.loc[df[column_name] == value]
 else:
    return df.loc[df[column_name] != value]

def find_occurences(df, antecedents, consequent, pos):
 '''
 Find the number of rows that match the antecedent and consequent, with
 pos being a list of conditions
 that should be inverted
 '''
 column_names: list = []
 for name in df.columns:
    column_names.append(str(name))

 index = 0
 df_new = df.copy()
 for antecedent in antecedents:
    value = antecedent[2]
    if index in pos:
        df_new = compare(df_new, column_names[antecedent[0]], int_to_operator_opposite[antecedent[1]], value)
    else:
        df_new = compare(df_new, column_names[antecedent[0]], int_to_operator[antecedent[1]], value)
        index += 1
 
 value = consequent[2]
 if -1 in pos:
    df_new = compare(df_new, column_names[consequent[0]], int_to_operator_opposite[consequent[1]], value)
 else:
    df_new = compare(df_new, column_names[consequent[0]], int_to_operator[consequent[1]], value)
 
 return len(df_new.index)

def prune(df, initial_error, Y1, E1, antecedents, consequent, confidence_level):
 '''
 Function to prune the rule based on error calcs, will call itself
 '''
 index = 0
 lowest_error = initial_error
 best_antecedents = antecedents.copy()
 for antecedent in antecedents:
    Y2 = find_occurences(df, antecedents, consequent, [index])
    E2 = find_occurences(df, antecedents, consequent, [index, -1])
    error = 1-proportion_confint(Y1+Y2, Y1+Y2+E1+E2, confidence_level, method='beta')[0] 
    if error < lowest_error:
        new_antecedents = antecedents.copy()
        lowest_error = error
        best_antecedents = new_antecedents
        new_antecedents.remove(antecedent)
        new_Y1 = find_occurences(df, new_antecedents, consequent, [])
        new_E1 = find_occurences(df, new_antecedents, consequent, [-1])
        new_initial_error = 1-proportion_confint(Y1, E1+Y1, confidence_level, method='beta')[0]
        new_antecedents, _ = prune(df, new_initial_error, new_Y1, new_E1, new_antecedents, consequent, confidence_level)
    index += 1
 return [best_antecedents, consequent], lowest_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', help="Specify the name of the data file", default="..\data\credit_output_final_binned.csv")
    parser.add_argument('--rules_file', help="Specify the name of the rules file", default="rules.txt")
    parser.add_argument("--rule", help="Specify a custom rule to be run and pruned", default="")
    parser.add_argument('--type', help="Specify the type of operation to be run", choices=["PRUNE", "RULES"], default="PRUNE")
    args = parser.parse_args()
    
    # Modify this to run tests at a different alpha
    confidence_level = 0.75

    if args.type == "RULES":
        '''
        This generates rules in the specified format, given a weka format
        decision tree
        '''
        rules = open(args.rules_file, "r")
        df = pd.read_csv(args.data_file)
        generate_rules(df, rules)
        print("Boo")
    else:
        '''
        This runs through all the rules provided, and tries to prune them
        '''
        df = pd.read_csv(args.data_file)
        rules = []
        if args.rule == "":
            ruleFile = open(args.rules_file, "r")
            rules = ruleFile.readlines()
        else:
            rules.append(args.rule)

        antecedents = []
        consequent = []

        for rule in rules:
            print("-----------Rule:")
            print(rule)
            print()
            print(f"Trying to prune with confidence level {1-confidence_level}")
            rule = ast.literal_eval(rule)

            antecedents = rule[0]
            consequent = rule[1]
            Y1 = find_occurences(df, antecedents, consequent, [])
            E1 = find_occurences(df, antecedents, consequent, [-1])

            print(Y1)
            print(E1)

            if(E1 == 0 and Y1 == 0):
                print("Could not find any occurences of this rule in the data set!")
                print()
                continue
            
            initial_error = 1-proportion_confint(Y1, E1+Y1, confidence_level, method='beta')[0]
            better_rule, new_error = prune(df, initial_error, Y1, E1, antecedents, consequent, confidence_level)
            
            if(new_error != initial_error):
                print(f"Pruned rule: {better_rule} improved error from {initial_error} to {new_error}")
            else:
                print(f"Found no better version, rule still remains: {better_rule}") 
                pass
            print()