# import the necessary packages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functions import *
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("\n===========================================\n")

print("> Libraries Imported")

"""
SETTINGS
"""

gallery_path = "../dataset/test"
query_path = "../dataset/test/query"
setup_dir_path = "setup_files"

# ResNet Settings
img_size = 512
pooling = "max"
num_epochs = 30

# ImageDataGenerator Settings
batch_size = 64

# NearestNeighbors Settings
algorithm_for_NN = "ball_tree"
metric_for_NN = "manhattan"

"""
SCRIPT
"""

# Create `Resnet50` model without top layers so we get convolutional features as output instead of the image class probability.

model = ResNet50(weights="imagenet", include_top=False,
                 input_shape=(img_size, img_size, 3), pooling=pooling)
print("\n> Model ResNet50 Imported")

img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

datagen = img_gen.flow_from_directory(gallery_path,
                                      classes=['gallery'],
                                      target_size=(img_size, img_size),
                                      batch_size=batch_size,
                                      class_mode=None,
                                      shuffle=False)

feature_list = model.predict(datagen, num_epochs)

# Get full path for all the images in our dataset
filenames = [gallery_path + "/" + s for s in datagen.filenames]

print("\n> Fitting Nearest Neighbor Algorithm")
neighbors = NearestNeighbors(n_neighbors=10,
                             algorithm=algorithm_for_NN,
                             metric=metric_for_NN)
neighbors.fit(feature_list)

# Execute query predictions
print("\n> Executing Query")
queryImageAll(query_path, img_size, model, neighbors, filenames)
print("\n> Results correctly stored")
