# import the necessary packages
from functions import *


# SETTINGS
gallery_path = "../dataset/training_augm"
gallery_features_path = "gallery_features.pkl"
query_path = "../dataset/validation/query"
query_features_path = "query_features.pkl"

# Get the features of all the gallery images
print("\n> Extracting gallery features")
    
# A. Extracting features
gallery_features = feature_extraction(gallery_path)
print("> Gallery features extracted")

# B. Saving extracted features
save(gallery_features, gallery_features_path)
print("> Gallery features saved")


# Get the features of all the query images
print("\n> Extracting query features")
    
# A. Extracting features
query_features = feature_extraction(query_path)
print("> Query features extracted")

# B. Saving extracted features
save(query_features, query_features_path)
print("> Query features saved\n")

res_df = queryImages(gallery_features, query_features)

print("\n> Results correctly stored")