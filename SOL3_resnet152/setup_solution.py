# import the necessary packages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib as pickle
from functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("\n===========================================\n")

print("> Libraries Imported")

"""
SETTINGS
"""

gallery_path = "../dataset/validation"
setup_dir_path = "setup_files"

# ResNet Settings
img_size = 256
pooling = "avg"
num_epochs = 20

# ImageDataGenerator Settings
batch_size = 128

# NearestNeighbors Settings
algorithm_for_NN = "ball_tree"
metric_for_NN = "euclidean"

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


# Save ResNet model, NearestNeighbors model and filenames

print("\n> Saving ResNet model, NearestNeighbors model and filenames\n")

model_resnet50_path = setup_dir_path + "/model_resnet50.sav"
model_neighbors_path = setup_dir_path + "/model_neighbors.sav"
filenames_index_path = setup_dir_path + "/filenames_index.sav"

model.save(model_resnet50_path)
pickle.dump(neighbors, open(model_neighbors_path, 'wb'))
pickle.dump(filenames, open(filenames_index_path, 'wb'))

print("\n> All saved.")
