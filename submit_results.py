#%%
import requests
import json
import os
import pandas as pd
import time

"""
SETTINGS
"""
#%%
results_path = 'tools/model_evaluation/test_11_final_test_venv'

url_unitn_vpn = "http://kamino.disi.unitn.it:3001/results/"
url_aws_test = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/test/"
url_aws_results = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/results/"

url_chosen = url_aws_test

"""
FUNCTIONS
"""
#%%
# Professor testing
def load_results():
    mydata = dict()
    mydata['groupname'] = "Dynamic Trio"
    res = dict()
    # res['<query image name>'] = ['<gallery image rank 1>', '<gallery image rank 2>', ..., '<gallery image rank 10>']
    res["img001.jpg"] = ["gal999.jpg", "gal345.jpg", "gal268.jpg", "gal180.jpg", "gal008.jpg", "gal316.jpg", "gal423.jpg", "gal111.jpg", "gal234.jpg", "gal730.jpg"]
    res["img002.jpg"] = ["gal336.jpg", "gal422.jpg", "gal194.jpg", "gal644.jpg", "gal910.jpg", "gal108.jpg", "gal179.jpg", "gal873.jpg", "gal556.jpg", "gal692.jpg"]
    res["img003.jpg"] = ["gal098.jpg", "gal879.jpg", "gal883.jpg", "gal556.jpg", "gal642.jpg", "gal329.jpg", "gal305.jpg", "gal068.jpg", "gal080.jpg", "gal018.jpg"]
    mydata["images"] = res
    submit(mydata)
    
# Submitting results (as json) to the server of unitn
# REMEMBER TO ACTIVATE GLOBALPROTECT (VPN)
def submit(results, solution_name):
    mydata = dict()
    mydata["groupname"] = "Dynamic Trio"
    mydata["images"] = results
    res = json.dumps(mydata)
    with open(results_path+'/'+solution_name+'.json', 'w') as f:
        json.dump(mydata, f)
    response = requests.post(url_chosen, res)
    result = json.loads(response.text)
    print(f"> Accuracy is {result['results']}")

# Convert the given dataframe to a json format, 
# where the first column is the set of keys and the remaining
# ones are saved inside as a list  
def convert_df_to_dict(df):
    res = dict()
    for _, row in df.iterrows():
        res[row['Query']+".jpg"] = [x+".jpg" for x in row[1:]]
    return res

# Importing all dataframes contained in the specified path
def import_results(path = results_path):
    results = []
    names = []
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                results.append(pd.read_csv(os.path.join(path,file)))
                names.append(file)
    return results,names
    
# Import all results, converts them into json and then sends them all
def load_all_results():
    dfs,dfs_names = import_results()
    jsons = [convert_df_to_dict(df) for df in dfs]

    tot_sols = len(jsons)
    for idx, solution in enumerate(jsons):
        name = dfs_names[jsons.index(solution)][:-4]
        print("-"*60)
        print("> Submitting " + name)

        submit(solution, name)
        if idx+1 < tot_sols:
            print("> Waiting 15 seconds for AWS")
            time.sleep(15)
    print("-"*60)
    print("> All results submitted")


"""
SCRIPT
"""
#%%
dfs,dfs_names = import_results()

# test on singular dataset
#dfs[0]
#json_test = convert_df_to_dict(dfs[0])
#json_test["assisi_church13.jpg"]

#%%
load_all_results()


# %%
#load_results()
# %%
