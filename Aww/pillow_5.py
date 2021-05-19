from PIL import Image
import glob
import os
import sys
import numpy

class Fingers():
    def __init__(self, image_in, person_in, sex_in, handedness_in, finger_type_in):
        self.im = image_in
        self.person = person_in
        self.sex = sex_in
        self.handedness = handedness_in
        self.finger_type = finger_type_in

imgs = []
for infile in glob.glob("*.BMP"):
    #Opens the file, name=file, location=im.
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    img_dtls = file.split('_')
    imgs.append(Fingers(im, img_dtls[0], img_dtls[2],img_dtls[3],img_dtls[4]))

n = 2
print(imgs[n].person, imgs[n].sex, imgs[n].handedness, imgs[n].finger_type)
print(imgs[0].im)
imgs[0].im.show()

    

