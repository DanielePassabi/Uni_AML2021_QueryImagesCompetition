#%%
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from pathlib import Path
from PIL import Image
# See https://keras.io/api/applications/ for details
#%%
class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, 
                           outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
    


#%%
fe = FeatureExtractor()
features = []
img_paths = []

#%%
from os import listdir
from os.path import isfile, join
for img_path in listdir("originalImages/"):
    if isfile(join("originalImages/",img_path)):
        # Extract Features
        feature = fe.extract(img=Image.open(join("originalImages/images",img_path)))
        # Save the Numpy array (.npy) on designated path
        feature_path = "originalImages/features/"+img_path+".npy"
        np.save(feature_path, feature)
#%%
# Importing features of our images
for feature_path in Path("originalImages/features").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("originalImages/images") / (feature_path.stem))
features = np.array(features)

#%%
test_image = Image.open("originalImages/images/rio0.jpg")

#%%
query = fe.extract(test_image)
dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
ids = np.argsort(dists)[:10]  # Top 30 results
ordered_results = [img_paths[i] for i in ids]
# %%
def loadImages(paths):
    # return array of images

    loadedImages = []
    for image in paths:
        img = Image.open(image)
        loadedImages.append(img)

    return loadedImages
#%%

#%%
path = "originalImages/images/"

# your images in an array
imgs =loadImages(ordered_results)
#%%
import matplotlib.pyplot as plt

w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 2
rows = 5
for i in range(1, columns*rows+1):
    img = imgs[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
# %%
