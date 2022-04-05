import time
import cv2
import numpy as np
import os
from os import listdir
import scipy.io
 
#winSize = (32,64) #width time height
winSize = (128,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.0
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 128
hog_descript = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
 #copied code https://www.geeksforgeeks.org/how-to-iterate-through-images-in-a-folder-python/
# get the path/directory
folder_dir = "C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/new_test_set_ducks"
folder_dir2 = "C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/new_test_set_no_ducks"
dataset_new_test = [] #500 col x 757 rows, last one is a 1 or 0, 1 for a duck, the first 250 are ducks the last 250 are not

#index = 0
for images in os.listdir(folder_dir):
    
    # check if the image ends with png
    if (images.endswith(".png")):

        im = cv2.imread('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/new_test_set_ducks/' + images)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hog_im = hog_descript.compute(im)
        #print(index)
        #index = index + 1
        #print(hog_im.size) #756
        #hog_im = np.append(hog_im,1.0)
        #print(hog_im)
        dataset_new_test.append(hog_im)


index = 0
for images in os.listdir(folder_dir2):
    if (index < 1000):
        # check if the image ends with png
        if (images.endswith(".png")):

            im = cv2.imread('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/new_test_set_no_ducks/' + images)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            hog_im = hog_descript.compute(im)
            #print(index)
            #index = index + 1
            #print(hog_im.size) #756
            #hog_im = np.append(hog_im,0.0)
            #print(hog_im)
            dataset_new_test.append(hog_im)
    index = index + 1

#print(np.size(dataset_500))
#print(dataset_500)

#https://stackoverflow.com/questions/1095265/matrix-from-python-to-matlab#:~:text=It%20takes%20a%20list%20of%20lists%20and%20returns,file%2C%20open%20it%20in%20matlab%2C%20and%20execute%20it.


scipy.io.savemat('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/dataset_new_test.mat', mdict={'arr': dataset_new_test})
