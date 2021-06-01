#%%
import shutil, random, os
import glob
import re
import sys
import math

#%%
# Starting directory
dirpath_query = 'dataset/training'
dirpath_gallery = 'dataset/training'

# Destination directory
destDirectoryGallery = 'dataset/validation/gallery'
destDirectoryQuery = 'dataset/validation/query'

# Number of images to pick
n_images_gallery = 500
n_images_query = 50

# Pick images for the gallery
def pick_images_gallery():
    filenames = random.sample(os.listdir(dirpath_gallery), n_images_gallery)
    return filenames

# Copy images from a directory to another
def copy_images(dirpath, filenames,destDirectory):
    for i in range(len(filenames)):
        srcpath = os.path.join(dirpath, filenames[i])
        shutil.copy2(srcpath, destDirectory)

# Creation of the gallery
def create_gallery(destDirectory,n_images):
    filenames = pick_images_gallery()
    copy_images(dirpath_gallery, filenames,destDirectory)
    print("Moved "+str(len(filenames))+" images")

def pick_images_query(n_images=n_images_query):
    filenames = get_filenames_from_directory(destDirectoryGallery+"/*.jpg")
    radices = [get_radix(x) for x in filenames]
    categories = list(set(radices))
    categories.remove("distractor")
    images_per_cat = math.ceil(n_images/len(categories))
    categories = categories * images_per_cat    
    i=0
    for category in categories:
        if i < n_images:
            images_to_pick = [filename for filename in os.listdir(dirpath_query) if filename.startswith(category) and filename not in filenames]
            if len(images_to_pick) >= images_per_cat:
                images_picked = random.sample(images_to_pick, images_per_cat)
            else:
                images_picked = images_to_pick
            copy_images(dirpath_query, images_picked,destDirectoryQuery) # copying images
            i+=len(images_picked)
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
# pick_images_query()
# %%
def pick_stressful_gallery(how_many_per_cat=1):
    # Get categories
    filenames = get_filenames_from_directory(dirpath_gallery+"/*.jpg")
    radices = [get_radix(x) for x in filenames]
    categories = set(radices)
    categories.remove("distractor")
    i=0 
    for category in categories:
        if i <= n_images_gallery:
            # Pick one random image per category
            images_to_pick = [filename for filename in os.listdir(dirpath_gallery) if filename.startswith(category)]
            if how_many_per_cat > len(images_to_pick):
                images_picked = images_to_pick
            else:
                images_picked = random.sample(images_to_pick, how_many_per_cat)
            
            # Copying the image
            copy_images(dirpath_gallery, images_picked,destDirectoryGallery) 
            i+=len(images_picked)
    
    # Pick distractor images
    images_to_pick = [filename for filename in os.listdir(dirpath_gallery) if filename.startswith("distractor")]
    if len(images_to_pick) >= (n_images_gallery-i):
        images_picked = random.sample(images_to_pick, n_images_gallery-i)
    else:
        images_picked = random.sample(images_to_pick, len(images_to_pick))
    copy_images(dirpath_gallery,images_picked,destDirectoryGallery) 

#%%
pick_stressful_gallery(how_many_per_cat=3)
#%%
pick_images_query()
# %%
