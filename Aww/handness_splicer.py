from PIL import Image
import glob
import os
import sys
import numpy
import random#EXCHANGE /Users/glas4/Pictures/the_archive with own path throught the document
from pathlib import Path
#Creates folders
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#EXCHANGE /Users/glas4/Pictures/the_archive with own path throught the document
datadir = "/Users/glas4/Pictures/archive/SOCOFing/handness2"


str_path = "C:/Users/glas4/Pictures/archive/SOCOFing/crp"
data_dir = Path(str_path)

#Creates folder with folder to add images

createFolder(datadir + '/R')
createFolder(datadir + '/L')

for infile in data_dir.glob("*.BMP"):
    #Opens the file, name=file, location=im.
    base=os.path.basename(infile)
    file, ext = os.path.splitext(base)
    

    im = Image.open(infile)
    img_dtls = file.split('_')
    if img_dtls[3] == "Left":
        im.save(datadir + '/L/'+file+ext,"BMP")
    elif img_dtls[3] == "Right":
        im.save(datadir + '/R/'+file+ext,"BMP")
    else:
        print("Warning!")



    

