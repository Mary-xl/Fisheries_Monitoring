
'''
This is written for The Nature Conservancy Fisheries Monitoring Competition at
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/
Author: Mary Li
'''

import numpy as np
import os
import shutil

def data_split(root):

    np.random.seed(2019)
    root_all=os.path.join(root,'train')
    root_train=os.path.join(root,'train_split')
    if not os.path.exists(root_train):
        os.mkdir(root_train)
    root_val=os.path.join(root,'val_split')
    if not os.path.exists(root_val):
        os.mkdir(root_val)

    FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    split_propertion=0.8

    num_train=0
    num_val=0
    for fish in FishNames:
        if fish not in os.listdir(root_train):
            os.mkdir(os.path.join(root_train,fish))
        if fish not in os.listdir(root_val):
                os.mkdir(os.path.join(root_val,fish))

        class_images=os.listdir(os.path.join(root_all,fish))
        np.random.shuffle(class_images)
        num_ctrain=int(len(class_images)*split_propertion)

        train_images=class_images[:num_ctrain]
        val_images=class_images[num_ctrain:]

        for img in train_images:
            source=os.path.join(root_all,fish,img)
            destiny=os.path.join(root_train,fish,img)
            shutil.copy(source,destiny)
            num_train+=1

        for img in val_images:
            source=os.path.join(root_all,fish,img)
            destiny=os.path.join(root_val,fish,img)
            shutil.copy(source,destiny)
            num_val+=1

    print ("Finish spliting the images into training set:{}, and validation set:{}".
           format(num_train,num_val))

    return (num_train, num_val)