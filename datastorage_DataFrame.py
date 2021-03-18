#!/usr/bin/env python
# coding: utf-8

# # Method for storing the dataset
# Store as pandas DataFrame

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

DATADIR = "/home/hanna/Documents/KEX/SoCoF/archive/SOCOFing/Altered/Altered-Hard" # directory to collect files from
IMG_SIZE = 90
FILE_NAME = "images_altered_hard.pkl" # Name of the file in which the data will be saved. .pkl if pickle, .csv if csv


# In[11]:


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


# In[12]:


def get_attributes(img):
    split_img = img.split('_')
    idty = int(split_img[0])
    gend = feat_dict[split_img[2]]
    hand = feat_dict[split_img[3]]
    fing = feat_dict[split_img[4]]
    return idty,gend,hand,fing


# In[22]:


def create_training_data():
    training_data = []
    for img in os.listdir(DATADIR):
        img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = new_array.flatten()
        idty,gend,hand,fing = get_attributes(img)
        training_data.append([new_array,idty,gend,hand,fing])
    return(training_data)


# In[23]:


training_data = create_training_data()


# In[6]:


print(len(training_data))


# In[7]:


# Store data in pandas DataFrame
df = pd.DataFrame(training_data, columns=["Image","Identity","Gender","Hand","Finger"])


# In[8]:


df.head()


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
df.to_csv(FILE_NAME,index=False)


# In[9]:


# Read file and store the data in DataFrame df
uncsvd_df = pd.read_csv('images2.csv')


# In[10]:


type(uncsvd_df["Image"][0])


# ## Pickle

# $+$ <b>Retains types</b> <br>
# $+$ Less disk space <br>
# $+$ Faster <br>
# $-$ Not human readable <br>
# $-$ Only Python <br>
# (https://stackoverflow.com/questions/48770542/what-is-the-difference-between-save-a-pandas-dataframe-to-pickle-and-to-csv)

# In[9]:


# Save data as pickle
df.to_pickle(FILE_NAME)


# In[12]:


# Read pickle data
unpickled_df = pd.read_pickle("images.pkl")


# In[13]:


type(unpickled_df["Image"][0])


# In[ ]:




