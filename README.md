# AML Competition 2021
___

This project aims to create an algorithm able to match the images in a query set with the images in a much larger set called gallery. It can be defined as a search engine for images. 

## Create a virtual environment

It is highly suggested to create a virtual environment and to install the packages contained inside `requirements.txt`. In order to create it, execute from command line:

```
virtualenv venv
```
Once created, it is necessary to activate the environment:

- in Unix systems:
    ```
    source venv
    ```
- in Windows systems:
    ```
    venv\Scripts\activate.bat
    ```

To install all the requirements:

```
pip install -r requirements.txt
```

## Structure of the project

The following sections will describe the structure of the folders inside the project.

### Dataset

The dataset folder contains:

- training folder with the initial images;
- training_augm, with the initial dataset and the augmented version of them;
- validation, which is empty and will be filled when needed with personalized or random gallery and query set. 

`create_subfolders.py` is a script for inserting all images with the same radix name (i.e. same category) inside the same folder, named after the radix name itself. 

`no_subfolders.py` makes the exact opposite: given a folder with subfolders, it brings all images out and it erases all the remaining empty folders.

`rename.py` was necessary at the start, to rename all images after the name of the directory. In fact, all images contained inside a folder were initially classified with the name of the location of the building. 

### Tools

The tools directory contains:

- competition results;
- `data_augmentation.ipynb`, for creating new images from the original ones, by rotating, shifting and increasing/decreasing the brightness and the zoom;
- model evaluation contains all the csv files with the results of various testing. In particular, it contains two interesting scripts: `model_evaluation.py` and `visual_model_evaluation.py`.

**Model evaluation** passes throughall the csv files inside a folder and evaluates the accuracy of the model: for each query, the accuracy is 1 if there is at least one match in the top k results. The algorithm provides this accuracy in terms of top 1, 3 and 10. Moreover, it returns a dataframe with the completely wrong matches in the top 10, so we can better understand the motivations underlying the model.

_For instance, in some cases, the model confused two similar palaces because the garden in the query image was missing, while there was in the gallery)._

**Visual model evaluation** takes all the csv files inside a folder and creates a visual representation of the results for each of them, providing a grid with the query image and the top 10 matching images from the gallery. 

_Note that the absolute path of images is necessary, since HTML won't find them otherwise. This means that, if the user wants to execute the algorithm by him/herself, he/she should change the path, until he/she reaches the training folder described above._

### Solutions

For each solution, there are two scripts, plus eventual additional files:

- `final_script_sol_k.py` can be executed to obtain a csv file with the results, stored in `tools/model_evaluation`;
- `functions.py` contains the code behind the brief solution contained in the previous file.

#### Solution 1: Match descriptors

This first solution is based on two classes and two functions:

- ColorDescriptor, which creates a feature vector of images, dividing each of them into five main areas: the four corners and an elliptical form of the center;
- Searcher, which compares the feature extracted from the query image with the ones already analyzed from the gallery (and saved in `gallery_features.csv`);
- `extractFeaturesFromGallery()` creates a descriptor and extracts the features for all images in the gallery, then saved;
- `queryImage()` extracts the feature from the single query image and compares it with all the gallery feature. Then `queryAllImages()` performs this function for all query images. 

This solution is quite simple and might be helpful for small datasets, but whenever the number of images in the gallery increases (let's say over 1,000), it takes too long to finish, because it has to compare each query image with all those contained in the gallery set. 

#### Solution 2: Match descriptors with different descriptor

This solution is similar to the previous one, with some differences:

- it saves the features of query and gallery in pickle format and then performs the search;
- the descriptor is simpler, since it converts all images to the same color space and then computes the histogram for each of them. 

Despite it is simpler than the previous one, this solution tends to perform better, but still it is time consuming for big datasets. 

#### Solution 3: ResNet50

First of all, a pretrained ResNet50 model is created and then trained on the gallery set. Secondly, a K-nearest neighbour model is used to find the 10 closest images to the query set. This search is conducted through `queryImageAll()`, which:

1. Imports all query images;
2. Preprocess them and makes the ResNet model predict them;
3. Finds, for each image, the 10 most similar images by making a prediction through the K-nearest neighbour model. 

This appears to be the best solution among all those proposed, despite the following one gives similar results in terms of accuracy. 

#### Solution 4: ResNet152

This solution is similar to the previous one,except for the ResNet model chosen: we decided to give a try to ResNet152, which has more layers, thinking it would have provided better results. Actually, they are quite resembling the antecedent model. 

#### Solution 5:

The only missing tassel to the past two solutions is the training step. Since ResNets are pretrained models, we did not trained them on the training dataset of images. This solution tries to train the model for completeness, since results are reached till now are quite high, and to make our model more accurate on building recognition. 

Unfortunately, the model performs worst than the past two, both in terms of accuracy and computational time. 

### Additional scripts

The remaining two scripts in the main folder are:

- `select_random_images.py`: this script randomly generates the gallery and query folder from the starting training set.
- `submit_results.py`: script for converting csv files of matchings to a json format, which is requested to be sent in order to submit competition results. 

#### Select random images

In order to create the gallery, `create_gallery()` simply selects randomically images and copies them in the gallery folder. The alternative to this simple implementation is `pick_stressful_gallery()`, which selects a specified number of images for each category and then inserts the remaining images to select from the distractor class. In this way, we can see whether the model recognizes or not a limited set of related images from a big set of distractors. 

THe query is created through `pick_images_query()`, which takes, for each category one or more images, such that they do not belong to the distractor class and they do not show in the gallery class. 

This algorithm revealed to be useful for evaluating and comparing all the solutions proposed. 