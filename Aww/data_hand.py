import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

datadir = "/Users/glas4/Pictures/the_archive/full_collection"
IMG_SIZE = 90

class Fingerprint():
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
    
    def __init__(self):
        self.data = []
        self.identity = []
        self.gender = []
        self.hand = []
        self.finger = []
        
    def create_training_data(self, DATADIR, IMG_SIZE):
        training_data = []
        for img in os.listdir(DATADIR):
            img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            new_array = new_array.flatten() # Flatten the array
            idty,gend,hand,fing = self.get_attributes(img)
            self.data.append(new_array)
            self.identity.append(idty)
            self.gender.append(gend)
            self.hand.append(hand)
            self.finger.append(fing)

    def get_attributes(self,img):
        split_img = img.split('_')
        idty = int(split_img[0])
        gend = self.feat_dict[split_img[2]]
        hand = self.feat_dict[split_img[3]]
        fing = self.feat_dict[split_img[4]]
        return idty,gend,hand,fing

fingerprints = Fingerprint()
fingerprints.create_training_data(DATADIR, IMG_SIZE)