from PIL import Image
import glob
import os
import sys
import numpy

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
    im_rot90 = im.rotate(45,fillcolor = white)
    im_rot180 = im.rotate(135, fillcolor = white)
    im_rot270 = im.rotate(290,fillcolor = white)

    #Saves in seperate folders
    """
    im.save(datadir + '/cropped/'+file+'_rot0'+ext,"BMP")
    im_rot90.save(datadir + '/rotated90/'+file+'_rot90'+ext,"BMP")
    im_rot180.save(datadir + '/rotated180/'+file+'_rot180'+ext,"BMP")
    im_rot270.save(datadir + '/rotated270/'+file+'_rot270'+ext,"BMP")
    """
    #Saves in a single folder
    im.save(datadir + '/full_collection/'+file+'_rot0'+ext,"BMP")
    im_rot90.save(datadir + '/full_collection/'+file+'_rot90'+ext,"BMP")
    im_rot180.save(datadir + '/full_collection/'+file+'_rot180'+ext,"BMP")
    im_rot270.save(datadir + '/full_collection/'+file+'_rot270'+ext,"BMP")


    

