#%%
from functions import *
import cv2
import numpy as np
#%%
cd = ColorDescriptor((8,12,3))

#%%
im1 = cv2.imread("assisi_church1.jpg")
im2 = cv2.imread("assisi_church2.jpg")
im3 = cv2.imread("assisi_monaster6.jpg")
#%%
f1= cd.describe(im1)
f2 = cd.describe(im2)
f3 = cd.describe(im3)
#%%s
def distance(v1,v2,type='chi'):
    if type=='chi':
        return chi2_distance(v1, v2)
    else:
        return euclidean_distance(v1, v2)

# Euclidean distance for arrays (features)
def euclidean_distance(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sqrt(np.sum((v1-v2)**2))

# Chi Squared Distance for histograms
def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
    # return the chi-squared distance
    return d

#%%
distance(f1,f2,"euclidean")
# %%

