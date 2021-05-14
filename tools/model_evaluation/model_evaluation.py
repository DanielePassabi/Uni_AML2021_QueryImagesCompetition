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
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                results.append(pd.read_csv(file))
    return results

# %%
# Counts the matches for the single query
def count_matches(strings):
    i = 0
    for j in strings[1:]:
        if strings[0] == j:
            i+=1
    return round(i/(len(strings)-1),2)
#%%

# Evaluates the single model based on the df
def evaluate_model(df):
    
    # Divide strings and numbers
    r = re.compile("([a-zA-Z_]+)([0-9]+)") 
    matches = []
    for i in range(len(df)):
        query = df.iloc[i,:]
        query_cat = ([r.match(image).groups()[0] for image in query])
        matches.append(count_matches(query_cat))
    # Return the mean number of matches
    return np.mean(matches)
#%%
def evaluate_all_models():
    dfs = import_results()
    accuracy = []
    for df in dfs:
        accuracy.append(evaluate_model(df))
    return accuracy
#%%
evaluate_all_models()
