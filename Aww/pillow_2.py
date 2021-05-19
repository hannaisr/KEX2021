from PIL import Image
import glob, os
import sys
import numpy


file, ext = os.path.splitext("1__M_Left_little_finger.BMP")
im = Image.open("1__M_Left_little_finger.BMP")

(left, upper, right, lower) = (2, 9, 92, 99)

im = im.crop((left, upper, right, lower))
im = im.rotate(90)

im.save('/Users/glas4/Pictures/pic_test/save_folder/'+file+'_rot90'+'.bmp')

