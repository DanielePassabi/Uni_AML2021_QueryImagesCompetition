# import the necessary packages
from functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import joblib as pickle
from tensorflow import keras

print("\n===========================================\n")

print("> Libraries Imported")

"""
SETTINGS
"""

query_path = "../dataset/validation/query"
setup_dir_path = "setup_files"
img_size = 256

"""
SCRIPT
"""

print("\n> Loading all needed files for querying")

# Import ResNet model, NearestNeighbors model and filenames
model_resnet50_path = setup_dir_path + "/model_resnet50.sav"
model_neighbors_path = setup_dir_path + "/model_neighbors.sav"
filenames_index_path = setup_dir_path + "/filenames_index.sav"

model_resnet50 = keras.models.load_model(model_resnet50_path)
model_neighbors = pickle.load(open(model_neighbors_path, 'rb'))
filenames = pickle.load(open(filenames_index_path, 'rb'))

print("> All setup files loaded")

# Execute query predictions
print("\n> Executing Query")

queryImageAll(query_path, img_size, model_resnet50, model_neighbors, filenames)

print("\n> Results correctly stored")