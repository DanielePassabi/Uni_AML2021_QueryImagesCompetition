# import the necessary packages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from functions import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("\n===========================================\n")

print("> Libraries Imported")

"""
SETTINGS
"""

gallery_path = "../dataset/validation"
query_path = "../dataset/validation/query"
results_filename = "../tools/model_evaluation/SOL5_ResNet152_trained_results.csv"
training_path = '../dataset/training_augm_subfolders'

# ResNet Settings
img_size = 256
pooling = "avg"
num_epochs = 2
steps_epochs = 100
LOSS_METRICS = ['accuracy']
OBJECTIVE_FUNCTION = 'mse'
classes = 51
DENSE_LAYER_ACTIVATION = 'sigmoid'
# ImageDataGenerator Settings
batch_size = 128

# NearestNeighbors Settings
algorithm_for_NN = "ball_tree"
metric_for_NN = "euclidean"

"""
SCRIPT
"""

# Create `Resnet50` model without top layers so we get convolutional features as output instead of the image class probability.

model = keras.models.Sequential()
model.add(ResNet152(include_top = False, pooling = pooling, weights = 'imagenet',input_shape=(img_size, img_size, 3)))
model.add(keras.layers.Dense(classes, activation = DENSE_LAYER_ACTIVATION))
model.layers[0].trainable = False

print("\n> Model ResNet152 Imported")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n> Model Compiled")

print("\n> Loading images")
img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

datagen_training = img_gen.flow_from_directory(training_path,
                                      target_size=(img_size, img_size),
                                      batch_size=batch_size,
                                      class_mode="categorical",
                                      shuffle=False)

datagen_gallery = img_gen.flow_from_directory(gallery_path,
                                      classes=['gallery'],
                                      target_size=(img_size, img_size),
                                      batch_size=batch_size,
                                      class_mode="categorical",
                                      shuffle=False)

print("\n> Fitting the model on the augmented training set")
fit_history = model.fit(
        datagen_training,
        steps_per_epoch=steps_epochs,
        epochs=num_epochs,
        validation_data=datagen_gallery,
        validation_steps=100
    )

print("\n> Predicting on validation set")
feature_list = model.predict(datagen_gallery, num_epochs)

# Get full path for all the images in our dataset
filenames = [gallery_path + "/" + s for s in datagen_gallery.filenames]

print("\n> Fitting Nearest Neighbor Algorithm")
neighbors = NearestNeighbors(n_neighbors=10,
                             algorithm=algorithm_for_NN,
                             metric=metric_for_NN)
neighbors.fit(feature_list)

# Execute query predictions
print("\n> Executing Query")
queryImageAll(query_path, img_size, model, neighbors, filenames,results_filename)
print("\n> Results correctly stored")
