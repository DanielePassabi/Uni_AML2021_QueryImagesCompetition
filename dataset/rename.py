#%%
import os
path = 'validation/query/'
files = os.listdir(path)
#%%

#%%
for index,file in enumerate(files):
    temp_path = path+file+"/"
    for i,f in enumerate(os.listdir(temp_path)):
        os.rename(os.path.join(temp_path, f), os.path.join(temp_path, str(i).join([file,'.jpg'])))
# %%
