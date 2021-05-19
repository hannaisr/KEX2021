
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

#DATADIR = "/Users/glas4/Pictures/archive/SOCOFing/Real" # directory to collect files from
#FILE_NAME = "org_imgs_newcrop_REAL.pkl" # Name of the file in which the data will be saved. .pkl if pickle, .csv if csv

#DATADIR = "/Users/glas4/Pictures/archive/SOCOFing/Altered/Altered-Easy"
#FILE_NAME = "org_imgs_newcrop_EASY.pkl"

#DATADIR = "/Users/glas4/Pictures/archive/SOCOFing/Altered/Altered-Medium"
#FILE_NAME = "org_imgs_newcrop_MEDIUM.pkl"

DATADIR = "/Users/glas4/Pictures/archive/SOCOFing/Altered/Altered-Hard"
FILE_NAME = "org_imgs_newcrop_HARD.pkl"

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

def get_attributes(img_name):
    split_img = img_name.split('_')
    idty = int(split_img[0])
    gend = feat_dict[split_img[2]]
    hand = feat_dict[split_img[3]]
    fing = feat_dict[split_img[4]]
    return idty,gend,hand,fing

def crop_image(img_array):
    """Crops images to size 97x90.
    This is the optimal size for removing all unnecessary transparent
    areas but still keeping as much of the fingerprint as possible"""
    #print(img_array[0])
    new_array = np.delete(img_array, (0,1), axis=0)
    new_array = np.delete(new_array, slice(90,None), axis=0)
    new_array = np.delete(new_array, (0,1), axis=1)
    new_array = np.delete(new_array, slice(90,None),axis=1)
    return new_array

def create_training_data():
    training_data = []

    for img in os.listdir(DATADIR):

        img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE)
        #print(img_array)
        new_array = crop_image(img_array) # Should be outcommented if images are already cropped
        #new_array = new_array.flatten()
        idty,gend,hand,fing = get_attributes(img)
        training_data.append([new_array,idty,gend,hand,fing])
    
    return(training_data)

training_data = create_training_data()

print(len(training_data))
print(training_data[0])
# Store data in pandas DataFrame
df = pd.DataFrame(training_data, columns=["Image","Identity","Gender","Hand","Finger"])

# Save data as pickle
df.to_pickle(FILE_NAME)

