gallery_path = "../../dataset/validation/gallery"
training_path= "../../dataset/training"
training_path_augm = "../../dataset/training_augm"
query_path = "../../dataset/validation/query"
#%%
from keras.applications.resnet import ResNet152
from scipy.ndimage.interpolation import zoom
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale = 1.0/255.,
                               zoom_range=0.5,
                               horizontal_flip=True)

test_gen = ImageDataGenerator(rescale = 1.0/255.,
                              zoom_range=0.5,
                              horizontal_flip=True)

train_generator = train_gen.flow_from_directory(training_path,
                                                batch_size=20,
                                                class_mode='binary',
                                                target_size=(256,256))

validation_generator = train_gen.flow_from_directory(training_path,
                                                     batch_size=20,
                                                     class_mode='binary',
                                                     target_size=(256, 256))
#%%
#%%
from keras.applications import *

model = ResNet152()
preds = model.predict(prepro)
print(decode_predictions(preds))