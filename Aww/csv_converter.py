import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

DATADIR = "/Users/glas4/Pictures/the_archive/full_collection/"
IMG_SIZE = 100

feat_dict = {
    "M" : 0,
    "F" : 1,
    "Left" : 0,
    "Right" : 1,
    "thumb" : 0,
    "index" : 1,
    "middle" : 2,
    "ring" : 3,
    "little" : 4
    }

def get_attributes(img):
    split_img = img.split('_')
    idty = split_img[0]
    gend = feat_dict[split_img[2]]
    hand = feat_dict[split_img[3]]
    fing = feat_dict[split_img[4]]
    return idty,gend,hand,fing

def create_training_data():
    training_data = []
    for img in os.listdir(DATADIR):
        img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        idty,gend,hand,fing = get_attributes(img)
        training_data.append([new_array,idty,gend,hand,fing])
    return(training_data)

training_data = create_training_data()

# Store data in pandas DataFrame
df = pd.DataFrame(training_data, columns=["Image","Identity","Gender","Hand","Finger"])

# Save data to csv file "images.csv"
df.to_csv('images.csv',index=False)

