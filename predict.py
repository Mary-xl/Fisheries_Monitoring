'''
This is written for The Nature Conservancy Fisheries Monitoring Competition at
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/
Author: Mary Li
'''

import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def predict(model, root_dir, batch_size,num_epochs, num_test_samples, img_w, img_h,FishNames):

    weights_file= os.path.join(root_dir, 'result/weights.h5')
    test_data_dir = os.path.join(root_dir, 'test_stg1/')

    # test data generator for prediction
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        shuffle=False,  # Important !!!
        classes=None,
        class_mode=None)

    test_image_list = test_generator.filenames

    print('Loading model and weights from trained weights...')
    model = load_model(weights_file)

    print ('Begin predict...')
    predictions=model.predict_generator(test_generator, num_test_samples)
    np.savetxt(os.path.join(root_dir,'result/predictions.txt'),predictions)

    print('Begin to write submission file ..')
    f_submit = open(os.path.join(root_dir, 'result/submit.csv'), 'w')
    f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
    for i, image_name in enumerate(test_image_list):
        pred = ['%.6f' % p for p in predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, num_test_samples))
        f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

    f_submit.close()

    print('Submission file successfully generated!')