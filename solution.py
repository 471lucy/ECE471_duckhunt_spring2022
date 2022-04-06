import time
import cv2
import scipy.io
import numpy as np


#added os for getting data
import os
from os import listdir

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""

#https://stackoverflow.com/questions/40363277/import-matlab-cell-array-into-python-for-scikit-learn
#model_ws = scipy.io.loadmat('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/duckhunt_ws_better.mat')
#print(model_ws)

#new tested model

#odel_ws_1000= scipy.io.loadmat('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/Duck_hunt_model_ws.mat')
model_ws_64 = scipy.io.loadmat('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/model_ws_64.mat')


# for collecting data updating folder name
#folder_in = 1 #everytime get a fram updates this

# https://pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
# https://github.com/techwithtim/OpenCV-Tutorials/blob/main/tutorial7.py

def GetLocation(move_type, env, current_frame, folder_in): #added folder_in for data collection remove later
    time.sleep(1) #artificial one second processing time

    # Converting the frame to grey
    # Getting the shape of the game screen
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    #os.mkdir('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/training_images' +  str(folder_in)) #for collecting data
    frameHeight, frameWidth = current_frame.shape

    #the frame width is 1024
    #the frame height is 768
    #the bird is in an area that is around 64x32 WxH
    #divide the frame into subframes that are 64x32, for a total of 384 sub frames
    #find the hog of each subframe, or apply the hog to each sub frame and store in image of HOG's
    #frame = cv2.imread(current_frame)

    #https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html
    #https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
    #https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html

    #print(frameHeight, frameWidth) #frame height is 1024, frame width is 768, frame is on its side transposed
    winSize = (64,64) #width time height
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
    #initialize list of hog
    hog_subframes = []
    index = 0
    
    #cv2.imshow('frame',current_frame)
    #cv2.waitKey(0)
    #loc = (0,0)
    #loc = []


    loc = []
    loc_in = 0 #index indicating number of coordinates in loc, don't want to exceed 10
    
    

    class_compare = []
    loc_compare = []
    for i in range(128,frameHeight+1,128):
        for j in range(128,frameWidth+1,128):
            #print(current_frame[j-32:j,i-64:i])
            im = current_frame[i-128:i,j-128:j]
            
            #for collecting data ****
            im_gaussian = cv2.GaussianBlur(im,(5,5),0)
            im_resized = cv2.resize(im_gaussian, (64,64), interpolation = cv2.INTER_AREA)
            #cv2.imwrite('C:/myvenvs/Project/ece471_536-S2022-main/ece471_536-S2022-main/duck-hunt/training_images/image-' + str(folder_in) + str(index) + '.png', im_resized)


            index = index+1
            hog_subframe = hog_descript.compute(im_resized)
            hog_subframe = np.append(hog_subframe, 1.0)
            #classify = np.sign(np.dot(hog_subframe, model_ws_64.get("ws")))
            classify = np.dot(hog_subframe, model_ws_64.get("ws"))
            #print(classify)

            

            if classify > 0:
                
                
                loc.append((i-32,j-32))
                loc_in = loc_in + 1
                loc.append((i-32,j-96))
                loc_in = loc_in + 1
                loc.append((i-96,j-32))
                loc_in = loc_in + 1
                loc.append((i-96,j-96))
                loc_in = loc_in + 1

                #loc.append((i-64,j-64))
                #print(loc)
                # loc = loc.append((i-64,j-64))
                #loc.append((i-64,j-64))
                #print(loc)
            



            #if its empty coordinates should be 0,0
            #if not loc:
             #   loc.append((0,0))            
            
            #if j == 64:
            #    cv2.imshow('frame', current_frame[i-64:i,j-32:j])
            #    print(current_frame[i-32:i,j-64:j].shape)
            #    cv2.waitKey(0)
            #print(np.append(hog_descript.compute(im),1.0))
            #hog_subframes.append(np.append(hog_descript.compute(im),1.0))
            #hog_subframe = hog_descript.compute(im)
            #print("aaaaaaaaaaaaaaaaaaaaaaa",hog_subframe.shape)
            #hog_subframe = np.append(hog_subframe, 1.0) #for bias
            #hog_subframe = np.append(hog_subframe, 1.0) #
            #print("aaaaaaaaaaaaaaaaaaaaaaa",hog_subframe.shape)
            #index = index + 1
            #print(model_ws_1000.get("ws"))
            
            #classify = np.sign(np.dot(hog_subframes[index],model_ws_1000.get("ws"))[0])
            #classify = np.dot(hog_subframe, model_ws_1000.get("ws"))
            #print(classify)
            #print(classify)s
            #if classify > 0: #predicted a duck
                    #class_compare.append(classify)
                    #loc_compare.append((i-32,j-32)) #print(hog_subframes[index])
                    #loc.append((i-64,j-64))
            #        loc = ((i-64,j-64))
            #index = index + 1
            #print(index, i, j)
            #print(np.transpose(current_frame[i-32:i,j-64:j]))
            

        
    #https://datagy.io/python-index-of-max-item-list/
    #print(min(class_compare))
    #index_max = class_compare.index(min(class_compare))
    #loc = loc_compare[index_max]
    #print the results subframes to a files
    # show the subframes as an image
    # create a training data set manually where I identify bird positives and negatives
    # get the hog of the training data set
    # seperately train a model - get the w parameters for a binary classification
    # 
    # 
    # in the solution: apply the model to get a sign 1: for a bird, -1 for a no bird
    # send the coordinates that detect a bird
    # for the coordinates that don't detect a bird, send 0,0 






    #procedure: thoughts:
    #devide the image into cubes that could contain a duck
    #convert each frame into hog of itself (can use third party software for this?)
    #reduce size of img to make go faster?
    #use a trained model to detect : yes duck in this frame, no duck not in this frame
    #(going to have to train serperately?)
    #model is binary classification model - duck or no duck with grad descent or softmax regression...?
    #if yes duck, take center coordinates as the coordinates of the duck return this

    





    #loc = (0,0)
    

    #Use relative coordinates to the current position of the "gun", defined as an integer below
    if move_type == "relative": #dont need to implement a relative solution for now
        """
        North = 0
        North-East = 1
        East = 2
        South-East = 3
        South = 4
        South-West = 5
        West = 6
        North-West = 7
        NOOP = 8
        """
        coordinate = env.action_space.sample() 

    
    #Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (W, H) 
        """




        #if the coordinates are a list greater than 10 have to get rid of some
        while (len(loc) > 10):
            loc.pop()


        #for absolute the coordinates are the location of the duck found above
        coordinate = loc





        #print(coordinate)
    
    #return coordinates found and move type
    return [{'coordinate' : coordinate, 'move_type' : move_type}]

