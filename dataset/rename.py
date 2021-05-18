#%%
import os
path = 'C:/Users/danyp/Documents/Programmazione/AML_Competition_2021/AML_Competition_2021/dataset/temp/'
files = os.listdir(path)
#%%

current_distractors_number = 172

#%%
for index,file in enumerate(files):
    temp_path = path+file+"/"
    for i,f in enumerate(os.listdir(temp_path)):
        os.rename(os.path.join(temp_path, f), os.path.join(temp_path, str(i+current_distractors_number).join([file,'.jpg'])))
# %%
