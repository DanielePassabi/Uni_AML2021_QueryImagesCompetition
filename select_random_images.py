#%%
import shutil, random, os
import glob
import re
import sys
import math

#%%
# Starting directory
dirpath = 'dataset/training_augm'

# Destination directory
destDirectoryGallery = 'dataset/validation/gallery'
destDirectoryQuery = 'dataset/validation/query'

# Number of images to pick
n_images_gallery = 10000
n_images_query = 200

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
    categories = list(set(radices)) 
    images_per_cat = math.ceil(n_images/len(categories))
    print(len(categories))
    print(images_per_cat)
    categories = categories * images_per_cat
    print(len(categories))
    i=0
    for category in categories:
        if i < n_images:
            if not category.startswith("distractor"):
                images_to_pick = [filename for filename in os.listdir(dirpath) if filename.startswith(category)]
                if len(images_to_pick) >= images_per_cat:
                    images_picked = random.sample(images_to_pick, images_per_cat)
                else:
                    images_picked = images_to_pick
                copy_images(images_picked,destDirectoryQuery) # copying images
                i+=len(images_picked)
                print(i)
    print("Added "+str(i)+" in the query dir.")

#%%
def get_filenames_from_directory(directory):
    filenames = glob.glob(directory)
    return [x.split("\\")[-1] for x in filenames]

def get_radix(s):
    r = re.compile("([a-zA-Z_]+)([0-9]+)") 
    return r.match(s).group(1)

#%%
# Creation of the gallery
#create_gallery(destDirectoryGallery,n_images_gallery)
# %%
pick_images_query()
# %%
