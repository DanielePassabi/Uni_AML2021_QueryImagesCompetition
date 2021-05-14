import argparse
import glob
import os
import pickle
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
from scipy.spatial.distance import euclidean


################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 1: Image Descriptor
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
class RGBHistogram:
    """
    Image descriptor using color histogram.

    :param bins: list
        Histogram size. 
        1-D list containing ideal values between 8 and 128; 
        but you can go up till 0 - 256.

    Example:
        >>> histogram = RGBHistogram(bins=[32, 32, 32])
        >>> feature_vector = histogram.describe(image='folder/image.jpg')
        >>> print(feature_vector.shape)
    """

    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        """
        Color description of a given image

        - compute a 3D histogram in the RGB color space
        - then normalize the histogram so that images with the same content, 
          but either scaled larger or smaller will have (roughly) the same histogram

        :param image:
            Image to be described.
        :return: flattened 3-D histogram
            Flattened descriptor [feature vector].
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=self.bins, ranges=[0, 256] * 3)
        hist = cv2.normalize(hist, dst=hist.shape)
        return hist.flatten()


################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 2: Indexing
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
# key - image file name
# value - computed feature vector/descriptor
def feature_extraction(dataset):
    features = {}
    descriptor = RGBHistogram(bins=[8, 8, 8])

    if os.path.exists(dataset):

        for filename in tqdm(glob.glob(os.path.join(dataset, "*.jpg"))):

            # e.g. places/eiffel_tower.jpg => eiffel_tower
            img_name = os.path.basename(filename).split('.')[0]

            image = cv2.imread(filename)
            feature = descriptor.describe(image)
            # key - image name, value - feature vector
            features[img_name] = feature
        return features

    else:
        print("\n> ERROR: the specified directory [", dataset, "] does not exist.\n Aborting execution...")
        sys.exit(0)


# Writing the index to disk
def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    


################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 3: Searching
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
class Searcher:
    def __init__(self, features):
        self.features = features

    def search(self, query):
        results = {}

        for name, feature in self.features.items():
            #dist = euclidean(query, feature)
            dist = chi_squared(query, feature)
            results[name] = dist

        results = sorted([(d, n) for n, d in results.items()])
        return results




"""
Input:
    > gallery_features
    > query_features
"""
def queryImages(gallery_features, query_features):

    # Instantiate the searcher
    querySearcher = Searcher(gallery_features)

    res_list = []
    for key, value in tqdm(query_features.items()):

        res = querySearcher.search(value)[0:10]

        names_res = []
        for r in res:
            names_res.append(r[1])

        res_list.append([key] + names_res)
        res_df = pd.DataFrame(res_list, columns = ["Query", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        res_df.to_csv("../tools/model_evaluation/SOL2_results.csv", index=False)

    return res_df


"""
"""
def chi_squared(a, b, eps=1e-10):
    # compute the chi-squared distance
    dist = 0.5 * np.sum([pow(a - b, 2) / (a + b + eps)
                             for (a, b) in zip(a, b)])
    # return the chi-squared distance
    return dist