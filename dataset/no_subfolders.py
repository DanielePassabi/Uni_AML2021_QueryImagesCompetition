import shutil
import os
import glob

source_dir = 'validation_new'
target_dir = 'validation'
    
file_names = os.listdir(source_dir)

for file_name in file_names:
    for img in os.listdir(os.path.join(source_dir,file_name)):
        shutil.move(os.path.join(os.path.join(source_dir, file_name), img), target_dir)
