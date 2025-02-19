
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

DATADIR = "/Users/glas4/Pictures/archive/SOCOFing/Real" # directory to collect files from
DATADIR2 = "/Users/glas4/Pictures/archive/SOCOFing/Altered/Altered-Easy"
FILE_NAME = "org_imgs_newcrop.pkl" # Name of the file in which the data will be saved. .pkl if pickle, .csv if csv




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
    #print(gend)
    return idty,gend,hand,fing
"""
def crop_image(img_array):
    #print(img_array[0])
    new_array = np.delete(img_array, (0,1), axis=0)
    new_array = np.delete(new_array, slice(90,None), axis=0)
    new_array = np.delete(new_array, (0,1), axis=1)
    new_array = np.delete(new_array, slice(90,None),axis=1)
    return new_array
"""
def create_training_data():
    training_data = []
    i = 0
    for img in os.listdir(DATADIR):

        img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE)
        print(i)
        #print(img_array)
        #new_array = crop_image(img_array) # Should be outcommented if images are already cropped
        #new_array = new_array.flatten() #####################################################################COMMENT TO MAKE 3D matrix
        idty,gend,hand,fing = get_attributes(img)
        if gend == 0:
            print(i)
            i+=1
        #training_data.append([new_array,idty,gend,hand,fing])
    print(i)
    return(training_data)

training_data = create_training_data()
i = 0
#print(len(training_data))
#print(training_data[0])
# Store data in pandas DataFrame
df = pd.DataFrame(training_data, columns=["Image","Identity","Gender","Hand","Finger"])

# Save data as pickle
df.to_pickle(FILE_NAME)

