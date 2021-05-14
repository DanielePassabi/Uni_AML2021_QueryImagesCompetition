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
def count_matches(strings,k=10):
    i = 0
    for j in strings[1:(k+1)]:
        if strings[0] == j:
            i+=1
    return round(i/(len(strings)-1),2)
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
    # Return the mean number of matches
    return [np.mean(matches[0]),np.mean(matches[1]),np.mean(matches[2])]
#%%
def evaluate_all_models():
    dfs = import_results()
    accuracy = []
    for i in range(len(dfs)):
        accuracy.append(evaluate_model(dfs[i]))
        print("--- Solution "+str(i+1)+" ---")
        print("> Accuracy top 1: "+str(accuracy[i][0]))
        print("> Accuracy top 3: "+str(accuracy[i][1]))
        print("> Accuracy top 10: "+str(accuracy[i][2]))
    return accuracy
#%%
evaluate_all_models()
