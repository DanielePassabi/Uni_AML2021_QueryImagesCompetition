#%%
import shutil, random, os
import glob
import re
import sys

#%%
# Starting directory
dirpath = 'dataset/training_augm'

# Destination directory
destDirectoryGallery = 'dataset/validation/gallery'
destDirectoryQuery = 'dataset/validation/query'

# Number of images to pick
n_images_gallery = 2000
n_images_query = 100

# Pick images for the gallery
def pick_images_gallery():
    filenames = random.sample(os.listdir(dirpath), n_images_gallery)
    return filenames

# Copy images from a directory to another
def copy_images(filenames,destDirectory):
    for i in range(len(filenames)):
        srcpath = os.path.join(dirpath, filenames[i])
        shutil.copy2(srcpath, destDirectory)

# Creation of the gallery
def create_gallery(destDirectory,n_images):
    filenames = pick_images_gallery()
    copy_images(filenames,destDirectory)
    print("Moved "+str(len(filenames))+" images")

def pick_images_query(n_images=n_images_query):
    filenames = get_filenames_from_directory(destDirectoryGallery+"/*.jpg")
    radices = [get_radix(x) for x in filenames]
    radices = set(radices)
    i=0
    while i<n_images:
        image = random.sample(os.listdir(dirpath), 1)
        image_radix = get_radix(image[0])
        if image_radix in radices and image_radix != "distractor" and image not in os.listdir(destDirectoryGallery):
            copy_images(image,destDirectoryQuery)
            i+=1
    print("Added "+str(n_images)+" in the query dir.")

#%%
def get_filenames_from_directory(directory):
    filenames = glob.glob(directory)
    return [x.split("\\")[-1] for x in filenames]

def get_radix(s):
    r = re.compile("([a-zA-Z_]+)([0-9]+)") 
    return r.match(s).group(1)

#%%
# Creation of the gallery
create_gallery(destDirectoryGallery,n_images_gallery)
# %%
pick_images_query()
#%%
get_radix("yellow_building18_aug3.jpg")