# Importing the necessary libraries
import pandas as pd
from IPython.core.display import HTML
from tqdm import tqdm 
import glob

"""
SETTINGS
"""

# Set abs path as default in the function
res_dir = "test_10_simple_test_pre_competition"
images_abs_path = "C:/Users/auror/OneDrive/Documenti/GitHub/AML_Competition_2021/dataset/training_augm/"  # aurora
images_abs_path = "C:/Users/Daniele/Documents/Programmazione/Github/AML_Competition_2021/dataset/training_augm/"  # dany
images_abs_path = "C:/Users/elypa/AML_Competition_2021/dataset/training_augm/"  # ely


"""
FUNCTIONS
"""

def save_df_as_html_table(df, df_name, images_abs_path=images_abs_path):

    html_inj = '<img src="{path_img}" width="130" height="130" title="{img_name}"> <figcaption style="font-size: xx-small;">  {img_name}</figcaption>'

    for i, row in df.iterrows():

        path_img = images_abs_path + df.loc[i, "Query"] + ".jpg"
        img_name = clean_img_name(df.loc[i, "Query"])

        html_inj.format(path_img=path_img, img_name=img_name)

        # add html style to query col 
        df.loc[i, "Query"] = html_inj.format(path_img=path_img, img_name=img_name)

        # add html style to results cols 
        for n in range(10):
            path_img = images_abs_path + df.loc[i, str(n+1)] + ".jpg"
            img_name = clean_img_name(df.loc[i, str(n+1)])

            df.loc[i, str(n+1)] = html_inj.format(path_img=path_img, img_name=img_name)

        # save as html
        temp_name = df_name + ".html"
        df.to_html(temp_name, escape=False)


def get_visual_results(res_dir):
    for res_csv in tqdm(glob.glob(res_dir + "/*.csv"), "> Storing HTML results"):
        
        temp_df = pd.read_csv(res_csv)
        temp_name = clean_csv_name(res_csv)
        save_df_as_html_table(temp_df, res_dir + "/" + temp_name)


"""
Given a path, it returns ony the name of a .csv file
"""
def clean_csv_name(path):
    start = path.rfind('\\') +1
    return path[start:-4]

"""
Given a path, it returns ony the path after the last slash
"""
def clean_img_name(path):
    start = path.rfind('\\') +1
    return path[start:]

"""
SCRIPT
"""

get_visual_results(res_dir)