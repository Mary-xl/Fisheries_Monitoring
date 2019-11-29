
'''
This is written for The Nature Conservancy Fisheries Monitoring Competition at
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/
Author: Mary Li
'''

import os
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.layers import AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16


def build_model(img_w,img_h,use_imagenet=True):

    print("Loading model....")
    InceptionV3_bb=InceptionV3(include_top=False,weights='imagenet' if use_imagenet else None,
                                     input_tensor=None, input_shape=(img_w,img_h,3))
    # InceptionV3_bb=VGG16(include_top=False,weights='imagenet' if use_imagenet else None,
    #                                  input_tensor=None, input_shape=(img_w,img_h,3))
    print('Adding top layers....')
    output=InceptionV3_bb.get_layer(index=-1).output
    output=AveragePooling2D((8,8),strides=(8,8),name='avg_pool')(output)
    output=Flatten(name='flatten')(output)
    output=Dense(8,activation='softmax',name='predictions')(output)

    model=Model(InceptionV3_bb.input,output)
    print(model.summary())
    return model