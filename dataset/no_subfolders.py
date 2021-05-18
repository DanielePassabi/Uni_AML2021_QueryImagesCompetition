import shutil
import os
import glob

source_dir = 'new_data'
target_dir = 'new_data_no_sub'
    
file_names = os.listdir(source_dir)

for file_name in file_names:
    for img in os.listdir(os.path.join(source_dir,file_name)):
        shutil.move(os.path.join(os.path.join(source_dir, file_name), img), target_dir)
