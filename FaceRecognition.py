from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
import scipy
from scipy import ndimage
from inception_blocks_v2 import *
import scipy.ndimage
import sys

%matplotlib inline
%load_ext autoreload
%autoreload 2

np.set_printoptions(threshold=np.nan)

sys.path.insert(0,'F:\CSE\Machine Learning\Deep Learning Assignment\Face_Recognition')
sys.path.insert(0,'F:\CSE\Machine Learning\Deep Learning Assignment\Face_Recognition\images')

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())


FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)


database = {}
database["ankit"]=img_to_encoding("Ank-1.jpg",FRmodel)
database["adi"]=img_to_encoding("Adi-1.jpg",FRmodel)
'''
for x in range(2,33):
    database["ankit"]=np.vstack((database["ankit"],img_to_encoding("ank"+str(x)+".jpg",FRmodel)))
    database["arpit"]=np.vstack((database["arpit"],img_to_encoding("arpit"+str(x)+".jpg",FRmodel)))
database["arpit"].shape
'''


def verify(image_path, identity, database, model):
 
    encoding = img_to_encoding(image_path,model)
    
    for x in range(database[identity].shape[0]):
         dist = np.linalg.norm(encoding-database[identity][x])
         if dist<0.55:
             print("It's " + str(identity) + ", welcome home! "+str(x+1))
             door_open = True
             break
         else:
             print("It's not " + str(identity) + ", please go away")
             door_open = False
        
        
    return dist, door_open

verify("Adi-2.jpg", "ankit", database, FRmodel)




def who_is_it(image_path, database, model):
    
    encoding = img_to_encoding(image_path,model)

    min_dist = 100
  
    for (name, db_enc) in database.items():
        for x in range(database[name].shape[0]):
            dist = np.linalg.norm(encoding-db_enc[x])
            if dist<min_dist:
                min_dist = dist
                identity = name
            if min_dist > 0.55:
                print("Not in the database.")
                door_open=False
            else:
                print ("it's " + str(identity) + ", the distance is " + str(min_dist))
                door_open=True
        
    return min_dist, identity,door_open


who_is_it("Ank-2.jpg", database, FRmodel)


