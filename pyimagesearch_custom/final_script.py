# import the necessary packages
from functions import *
import pandas as pd

# SETTINGS
gallery_path = "../dataset/validation/gallery"
gallery_features_path = "gallery_features.csv"
query_path = "../dataset/validation/query"

# get the features of all the gallery image
print("\n> Extracting gallery features")
extractFeaturesFromGallery(gallery_path, gallery_features_path)
print("> Gallery features extracted")

print("\n> Finding queries")

# run queryImage on every image present in the gallery
res_df = queryImageAll(query_path)
print("> Results correctly stored")

model_scores = getModelScores(res_df)
print("\n> Model scores")
print("  - Top 1  Score:", round(model_scores[0], 2))
print("  - Top 3  Score:", round(model_scores[1], 2))
print("  - Top 10 Score:", round(model_scores[2], 2))