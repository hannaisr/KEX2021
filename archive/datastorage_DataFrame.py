#!/usr/bin/env python
# coding: utf-8

# # Method for storing the dataset
# Store as pandas DataFrame

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

DATADIR = "/home/hanna/Documents/KEX/SoCoF/archive/SOCOFing/Real" # directory to collect files from
FILE_NAME = "org_imgs_newcrop.pkl" # Name of the file in which the data will be saved. .pkl if pickle, .csv if csv


# In[2]:


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


# In[3]:


def get_attributes(img_name):
    split_img = img_name.split('_')
    idty = int(split_img[0])
    gend = feat_dict[split_img[2]]
    hand = feat_dict[split_img[3]]
    fing = feat_dict[split_img[4]]
    return idty,gend,hand,fing


# In[4]:


def crop_image(img_array):
    """Crops images to size 97x90.
    This is the optimal size for removing all unnecessary transparent
    areas but still keeping as much of the fingerprint as possible"""
    new_array = np.delete(img_array, (0,1), axis=0)
    new_array = np.delete(new_array, slice(97,None), axis=0)
    new_array = np.delete(new_array, (0,1), axis=1)
    new_array = np.delete(new_array, slice(90,None),axis=1)
    return new_array


# In[7]:


def create_training_data():
    training_data = []
    for img in os.listdir(DATADIR):
        img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE)
        new_array = crop_image(img_array) # Should be outcommented if images are already cropped
        new_array = new_array.flatten()
        idty,gend,hand,fing = get_attributes(img)
        training_data.append([new_array,idty,gend,hand,fing])
    return(training_data)


# In[8]:


training_data = create_training_data()


# In[9]:


print(len(training_data))


# In[10]:


# Store data in pandas DataFrame
df = pd.DataFrame(training_data, columns=["Image","Identity","Gender","Hand","Finger"])


# In[11]:


# df.head()


# In[12]:


# len(df["Image"][0])


# ## CSV

# $+$ Human readable <br>
# $+$ Works with other programs/programming languages <br>
# $-$ <b>Doesn't retain type</b> <br>
# $-$ Slower <br>
# $-$ More disk space <br>
# 
# (https://stackoverflow.com/questions/48770542/what-is-the-difference-between-save-a-pandas-dataframe-to-pickle-and-to-csv).

# In[8]:


# Save data to csv file "images.csv"
# df.to_csv(FILE_NAME,index=False)


# In[9]:


# Read file and store the data in DataFrame df
# uncsvd_df = pd.read_csv('images2.csv')


# In[10]:


# type(uncsvd_df["Image"][0])


# ## Pickle

# $+$ <b>Retains types</b> <br>
# $+$ Less disk space <br>
# $+$ Faster <br>
# $-$ Not human readable <br>
# $-$ Only Python <br>
# (https://stackoverflow.com/questions/48770542/what-is-the-difference-between-save-a-pandas-dataframe-to-pickle-and-to-csv)

# In[13]:


# Save data as pickle
df.to_pickle(FILE_NAME)


# In[12]:


# Read pickle data
# unpickled_df = pd.read_pickle("images.pkl")


# In[13]:


# type(unpickled_df["Image"][0])


# In[ ]:




