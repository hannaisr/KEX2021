from PIL import Image
import glob
import os
import sys
import numpy

class Fingers():
    def __init__(self, image_in, person_in, sex_in, handedness_in, index_in):
        self.im = image_in
        self.person = person_in
        self.sex = sex_in
        self.handedness = handedness_in
        self.index = index_in




file, ext = os.path.splitext("1__M_Left_little_finger.BMP")
im = Image.open("1__M_Left_little_finger.BMP")

img_dtls = file.split('_')

img = Fingers(im, img_dtls[0], img_dtls[2],img_dtls[3],img_dtls[4])

print(img.person, img.sex, img.handedness, img.index)



    

