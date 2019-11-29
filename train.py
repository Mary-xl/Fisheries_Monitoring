
'''
This is written for The Nature Conservancy Fisheries Monitoring Competition at
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/
Author: Mary Li
'''
import os
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def train_model(model, root_dir, batch_size,num_epochs,num_train_samples, num_val_samples, img_w, img_h,FishNames):

    train_dir=os.path.join(root_dir,'train_split')
    val_dir=os.path.join(root_dir,'val_split')
    result_dir=os.path.join(root_dir,'result')
    os.mkdir(result_dir)

    print ('data augmentation begins...')
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    # augmentation configuration validation: only rescaling
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        shuffle=True,
        classes=FishNames,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        shuffle=True,
        classes=FishNames,
        class_mode='categorical')

    print('model settings......')
    learning_rate = 0.0001
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # autosave best Model
    best_model_file = os.path.join(result_dir,'weights.h5')
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)

    print('training begins...')
    model.fit_generator(
        train_generator,
        samples_per_epoch=num_train_samples,
        nb_epoch=num_epochs,
        validation_data=validation_generator,
        nb_val_samples=num_val_samples,
        callbacks=[best_model])
