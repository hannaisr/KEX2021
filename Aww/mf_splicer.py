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
datadir = "/Users/glas4/Pictures/archive/SOCOFing/mf"


str_path = "C:/Users/glas4/Pictures/archive/SOCOFing/crp50"
data_dir = Path(str_path)

#Creates folder with folder to add images

createFolder(datadir + '/males')
createFolder(datadir + '/females')

for infile in data_dir.glob("*.BMP"):
    #Opens the file, name=file, location=im.
    base=os.path.basename(infile)
    file, ext = os.path.splitext(base)
    

    im = Image.open(infile)
    img_dtls = file.split('_')
    if img_dtls[2] == "M":
        im.save(datadir + '/males/'+file+ext,"BMP")
    elif img_dtls[2] == "F":
        im.save(datadir + '/females/'+file+ext,"BMP")
    else:
        print("Warning!")



    

