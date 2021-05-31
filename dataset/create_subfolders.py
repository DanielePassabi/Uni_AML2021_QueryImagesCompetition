from os.path import dirname
from tqdm import tqdm
import shutil, random, os
import glob
import os
import re

"""
SETTINGS
"""
# Starting directory
dirpath = 'training_augm'

# Destination directory
destDirectoryGallery = 'training_augm_subfolders/'



"""
FUNCTIONS
"""
"""
Given a path, it returns ony the name of a .jpg file
"""
def cleanName(path):
    clean_path = re.sub(r'\d+', '', path)
    clean_path = clean_path.replace("_aug","")
    start = clean_path.rfind('\\') +1
    return clean_path[start:-4]

"""
Copy images from a directory to another
"""
def copy_images(srcpath,destDirectory):
    shutil.copy2(srcpath, destDirectory)

"""
SCRIPT
"""

# iterate over all images present in training
for imagePath in tqdm(glob.glob(dirpath + "/*.jpg")):

    # get the clean name of the dir
    dirName = destDirectoryGallery + cleanName(imagePath) 
    
    # create a folder (if it does not exist)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    # copy the image in the subfolder
    copy_images(imagePath,dirName) 