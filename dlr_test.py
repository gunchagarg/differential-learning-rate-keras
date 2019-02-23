import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.applications.resnet50 import ResNet50
import pandas as pd
import numpy as np
import glob
import h5py
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Importing differential learning rate implementation on Adam optimizer
from dlr_implementation import Adam_dlr

# Specifying the layer names at which split is to be made
model_split_1 = 'res4a_branch2a'
model_split_2 = 'fc_start'

#included in our dependencies
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

train_generator=train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
                                                 
                                                 
base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), name = 'fc_start')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name = 'output')(x)  #num_classes: number of classes in the dataset

model = Model(inputs=base_model.input, outputs=[out])

# Extracting layers at which split is made
split_layer_1 = [layer for layer in model.layers if layer.name == model_split_1][0]
split_layer_2 = [layer for layer in model.layers if layer.name == model_split_2][0]

## Optional
## In case the we want to freeze the layers below the first split
# trainable = False
# for i,layer in enumerate(base_model.layers[:]):
#     try:
#         if layer.name == model_split_1:
#             trainable = True
#     except:
#         pass
#     layer.trainable = trainable
#     print(layer, layer.trainable)
# opt = Adam_dlr(split_l = [split_layer_1],
                 lr = [1e-4, 1e-2])

opt = Adam_dlr(split_l = [split_layer_1, split_layer_2], 
               lr = [1e-4, 1e-3, 1e-2])
               
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=["accuracy"])
                                                 
history = model.fit_generator(train_generator,
                              steps_per_epoch = train_generator.n//train_generator.batch_size,
                              epochs = 10)
                                    
