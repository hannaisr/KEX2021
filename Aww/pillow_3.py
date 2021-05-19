from PIL import Image
import glob, os
import sys
import numpy


file, ext = os.path.splitext("ERN.JPG")
im = Image.open("ERN.JPG")

(left, upper, right, lower) = (2, 9, 92, 99)

im = im.crop((left, upper, right, lower))
im = im.rotate(90)
print(im)
im.save("/Users/glas4/Pictures/pic_test/save_folder/"+file+".jpg","JPEG")

