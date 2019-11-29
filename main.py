'''
This is written for The Nature Conservancy Fisheries Monitoring Competition at
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/
Author: Mary Li
'''

from data_split import data_split
from build_model import build_model
from train import train_model
from predict import predict



if __name__=='__main__':

    root='/home/mary/AI/data/fish'
    FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    img_w=299
    img_h=299
    batch_size=32
    num_epochs=10
    num_test_samples=1000



    num_train, num_val=data_split(root)
    model=build_model(img_w, img_h)
    train_model(model, root, batch_size, num_epochs, num_train, num_val, img_w, img_h, FishNames)
    predict(model, root, batch_size, num_epochs, num_test_samples, img_w, img_h, FishNames)