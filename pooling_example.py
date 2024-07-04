
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import cv2

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os


#Hard Coding One Layer
#Input Layer
#Convolutional Layer --> mutliply to a filter to zoom in on region or randomize
#Nonlinearity --> Choose important features
#Pooling Layer --> Again, choose important fatures




#Input Layer
data_test = np.asarray([[0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0]])

data_test = np.reshape(1,8,8,1)

#Convolutional Layer/Nonlniearity
model_test = Sequential()
model_test.add(Conv2D(1, (2,2), input_size = (8,8,1), activation = "relu" ))

#Pooling Layer
model_test.add(MaxPooling2D())
'''
MaxPooling
[1,1]
[1,1]
[1,1]
[1,1]
[1,1] #2x5 layer for 4x4 input
           
'''

model_test.summary()

detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]

weights = [np.asarray(detector), np.asarray([0.0])]

# store the weights in the model
model_test.set_weights(weights) #weights determine the significance of the data (like a slope of a node)

# apply filter to input data
yhat = model_test.predict(data_test) #applying the pooling filter 
# enumerate rows
for r in range(yhat.shape[1]):
 # print each column in the row
 print([yhat[0,r,c,0] for c in range(yhat.shape[2])])