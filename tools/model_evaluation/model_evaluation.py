#%%
import pandas as pd
import os
import re
import numpy as np

#%%

# Import all csv files inside the specified path 
# (default value: current directory)
def import_results(path = os.path.abspath(os.getcwd())):
    results = []
    names = []
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                results.append(pd.read_csv(file))
                names.append(file)
    return results,names

# %%
# Counts the matches for the single query
def count_matches(strings,k=10):
    i = 0
    for j in strings[1:(k+1)]:
        if strings[0] == j:
            return 1
        elif strings[0] in j and "aug" in j:
            return 1
    return 0

def compute_total_match(matches):
    return round(sum(matches)/len(matches),5)

def find_worst(matches):
    indices = [i for i, x in enumerate(matches) if x == 0]
    return indices

def get_worsts(worsts,df,top=1):
    if top==1:
        return df.iloc[worsts[0],:]
    elif top==3:
        return df.iloc[worsts[1],:]
    else:        
        return df.iloc[worsts[2],:]


#%%
# Evaluates the single model based on the df
def evaluate_model(df):
    # Divide strings and numbers
    r = re.compile("([a-zA-Z_]+)([0-9]+)") 
    matches = [[],[],[]]
    for i in range(len(df)):
        query = df.iloc[i,:]
        query_cat = ([r.match(image).groups()[0] for image in query])
        matches[0].append(count_matches(query_cat,1))
        matches[1].append(count_matches(query_cat,3))
        matches[2].append(count_matches(query_cat,10))
    worsts = [find_worst(matches[0]),
              find_worst(matches[1]),
              find_worst(matches[2])]
    res = [compute_total_match(matches[0]),
           compute_total_match(matches[1]),
           compute_total_match(matches[2])]
    # Return the mean number of matches
    return res,worsts
#%%
def evaluate_all_models(print_wrong = False):
    dfs,dfs_names = import_results()
    wrong_matches = []
    accuracy = []
    for i in range(len(dfs)):
        res, worsts = evaluate_model(dfs[i])
        accuracy.append(res)
        print("\n--- Solution "+str(i+1)+": "+dfs_names[i]+" ---")
        print("> Accuracy top 1: "+str(accuracy[i][0]))
        print("> Accuracy top 3: "+str(accuracy[i][1]))
        print("> Accuracy top 10: "+str(accuracy[i][2]))
        if print_wrong==True:
            wrong_matches.append(get_worsts(worsts,dfs[i],top=1))
    return res,wrong_matches
#%%
res, wrong_matches = evaluate_all_models(print_wrong=True)

# %%
wrong_matches[0]