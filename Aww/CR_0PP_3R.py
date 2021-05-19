from PIL import Image
import glob
import os
import sys
import numpy
import random


#Creates folders
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#EXCHANGE /Users/glas4/Pictures/the_archive with own path throught the document
datadir = "/Users/glas4/Pictures/archive/SOCOFing"

folder_nme = '/crp50/'
#Creates folder with folder to add images
createFolder(datadir + folder_nme)
(left, upper, right, lower) = (22, 29, 72, 79) #50x50
#(left, upper, right, lower) = (2, 9, 92, 99) #90x90


for infile in glob.glob("*.BMP"):
    #Opens the file, name=file, location=im.
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    #Crops the image from 96x103 into 90x90
    im_crp = im.crop((left, upper, right, lower))
    im_crp.save(datadir + folder_nme+file+'_crp50'+ext,"BMP")