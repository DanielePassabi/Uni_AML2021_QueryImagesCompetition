# import the necessary packages
from functions import *
import glob
import pandas as pd

from tqdm import tqdm

# SETTINGS
gallery_path = "../dataset/validation/gallery"
gallery_features_path = "gallery_features.csv"
query_path = "../dataset/validation/query"

# get the features of all the gallery image
extractFeaturesFromGallery(gallery_path, gallery_features_path)


# FINAL SCRIPT
#res_list = []
#
## run queryImage on every image present in the gallery
#for imagePath in tqdm(glob.glob(query_path + "/*.jpg")):
#
#    tempRes = queryImage(imagePath)
#    res_list.append( [tempRes[0]] + tempRes[1])
#
#res_df = pd.DataFrame(res_list, columns =["Query", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
#
#res_df.to_csv("results.csv")

print(cleanName("cioa/test/nome.jpg"))