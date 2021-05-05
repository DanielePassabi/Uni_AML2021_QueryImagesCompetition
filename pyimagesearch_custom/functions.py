# import the necessary packages
import numpy as np
import pandas as pd
import cv2
import imutils
import glob
import csv
import re
from tqdm import tqdm

"""
Class to encapsulate all the necessary logic to extract the 3D HSV color histogram from the images
"""
class ColorDescriptor:

    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space and initialize the features used to quantify the image
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except:
            print("You can't see mee")
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # divide the image into four rectangles/segments (top-left, top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # construct an elliptical mask representing the center of the image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extract a color histogram from the elliptical region and update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the image,
        # using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])

        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()

        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()
            
        # return the histogram
        return hist


"""
Class to encapsulate all the necessary logic to compare the features of the query image to the features of the gallery image
"""
class Searcher:

	def __init__(self, galleryFeaturesPath):
		# store our galleary features path
		self.galleryFeaturesPath = galleryFeaturesPath

	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		results = {}

		# open the galleary features file for reading
		with open(self.galleryFeaturesPath) as f:

			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the galleary features
			for row in reader:

				# parse out the image ID and features, then compute the chi-squared distance 
                # between the features in our galleary features and our query features
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)

				# now that we have the distance between the two feature vectors, we can udpate the results dictionary
                #  - the key is the current image ID in the galleary features file
                #  - the value is the distance we just computed, representing how 'similar' the image in the galleary features file is to our query
				results[row[0]] = d
			
            # close the reader
			f.close()

		# sort our results, so that the smaller distances 
        # (i.e. the more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])

		# return our (limited) results
		return results[:limit]

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d

"""
Obtains the features of all the images present in the gallery and stores them in gallery_features.csv

Input:
    > gallery_path: directory path of the gallery
    > gallery_features_path: directory path of where the features will be stored
"""
def extractFeaturesFromGallery(gallery_path, gallery_features_path):

    # initialize the color descriptor
    cd = ColorDescriptor((8, 12, 3))

    # open the output index file for writing
    output = open(gallery_features_path, "w")

    # use glob to grab the image paths and loop over them
    for imagePath in glob.glob(gallery_path + "/*.jpg"):

        # extract the image ID (i.e. the unique filename) from the image path and load the image itself
        imageID = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath)

        # describe the image
        features = cd.describe(image)

        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))
        
    # close the index file
    output.close()



"""
Given a path image, queries it and finds the 10 most similar images in the gallery

Input: 
    > img_path: path of the image in the query gallery
Output
    > list with
      - path name of query image
      - list of 10 most similar images
      - distances of 10 selected images from the initial query image
"""
def queryImage(img_path):

    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))

    # load the query image and describe it
    query = cv2.imread(img_path)
    features = cd.describe(query)

    # perform the search
    searcher = Searcher("gallery_features.csv")
    results = searcher.search(features)

    distance_list = []
    predictions_list = []

    for r in results:
        distance_list.append(r[0])      # we save the distances, maybe we will need them
        predictions_list.append(cleanName(r[1]))

    return [cleanName(img_path), predictions_list, distance_list]


"""
Runs the function queryImage on all the images present in the given directory

Input:
    > query_path: directory with all the images to query
Output:
    > dataframe with best 10 predictions
"""
def queryImageAll(query_path):

    res_list = []
    for imagePath in tqdm(glob.glob(query_path + "/*.jpg")):
        tempRes = queryImage(imagePath)
        res_list.append( [tempRes[0]] + tempRes[1])

    res_df = pd.DataFrame(res_list, columns = ["Query", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    res_df.to_csv("results.csv", index=False)
    return res_df


"""
Given a path
"""
def cleanName(path):
    start = path.rfind('\\') +1
    return path[start:-4]

"""
Removes all the numbers from a given string
"""
def removeNumbers(string):
    pattern = r'[0-9]'
    clean_string = re.sub(pattern, '', string)
    return clean_string

"""
Provides the scores of a model 

Input:
    > csv with 10 predictions
Output:
    > list with
      - top1 score
      - top3 score
      - top10 score
"""
def getModelScores(res_df):

    # initialize matches
    matches_top1 = 0
    matches_top3 = 0
    matches_top10 = 0

    total_queries = len(res_df["Query"])

    for i in range(total_queries):

        # clean the query name
        temp_query = removeNumbers(res_df["Query"][i])

        # clean the predictions names
        temp_pred_1 = removeNumbers(res_df["1"][i])      
        temp_pred_2 = removeNumbers(res_df["2"][i])
        temp_pred_3 = removeNumbers(res_df["3"][i])
        temp_pred_4 = removeNumbers(res_df["4"][i])
        temp_pred_5 = removeNumbers(res_df["5"][i])
        temp_pred_6 = removeNumbers(res_df["6"][i])
        temp_pred_7 = removeNumbers(res_df["7"][i])
        temp_pred_8 = removeNumbers(res_df["8"][i])
        temp_pred_9 = removeNumbers(res_df["9"][i])
        temp_pred_10 = removeNumbers(res_df["10"][i])

        preds_list = [temp_pred_1, temp_pred_2, temp_pred_3, temp_pred_4, temp_pred_5, temp_pred_6, temp_pred_7, temp_pred_8, temp_pred_9, temp_pred_10]

        # case
        if temp_query == preds_list[0]:
            matches_top1 = matches_top1 + 1
        if temp_query in preds_list[0:3]:
            matches_top3 = matches_top3 + 1
        if temp_query in preds_list:
            matches_top10 = matches_top10 + 1

    # Calculate the 3 scores
    top1_score = matches_top1 / total_queries
    top3_score = matches_top3 / total_queries
    top10_score = matches_top10 / total_queries

    return[top1_score, top3_score, top10_score]