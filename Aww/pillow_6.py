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
datadir = "/Users/glas4/Pictures/the_archive"


#Creates folder with folder to add images
createFolder(datadir)
createFolder(datadir + '/full_collection')
"""
createFolder(datadir + '/cropped')
createFolder(datadir + '/rotated90')
createFolder(datadir + '/rotated180')
createFolder(datadir + '/rotated270')
"""

for infile in glob.glob("*.BMP"):
    #Opens the file, name=file, location=im.
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)

    #Crops the image from 96x103 into 90x90
    (left, upper, right, lower) = (2, 9, 92, 99)
    im = im.crop((left, upper, right, lower))
    white = (255,255,255)
    #Rotates the cropped image by 90 degrees
    for i in range(1,360):

        im_2 = im.rotate(i,fillcolor = white)
        im_2.save(datadir + '/full_collection/'+file+'_min'+str(i)+ext,"BMP")



    

