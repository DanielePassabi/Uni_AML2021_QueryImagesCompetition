# import the necessary packages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


"""
Query of a single image
"""
def queryImage(img_path, img_size, model_resnet50, model_neighbors, filenames):
    input_shape = (img_size, img_size, 3)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    test_img_features = model_resnet50.predict(preprocessed_img, batch_size=1)
    _, indices = model_neighbors.kneighbors(test_img_features)

    pred_names = []
    for index in indices[0]:
        pred_names.append(cleanName(filenames[index]))

    return pred_names


"""
Query of all images in a given directory
"""
def queryImageAll(query_path, img_size, model_resnet50, model_neighbors, filenames):

    res_list = []
    for img_path in tqdm(glob.glob(query_path + "/*.jpg")):
        tempRes = queryImage(img_path, img_size, model_resnet50, model_neighbors, filenames)
        res_list.append( [cleanName(img_path)] + tempRes)

    res_df = pd.DataFrame(res_list, columns = ["Query", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    res_df.to_csv("../tools/model_evaluation/SOL3_ResNet50_results.csv", index=False)
    return res_df


"""
Given a path, it terurns ony the name of a .jpg file
"""
def cleanName(path):
    start = path.rfind('\\') +1
    return path[start:-4]