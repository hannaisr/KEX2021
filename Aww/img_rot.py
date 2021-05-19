from scipy import ndimage, misc
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rnd

FROM_FILE = "original_images.pkl" # Pickle file to collect images from 
TO_FILE = "10_thumbs_expanded_dataset.pkl" # Name of file to store data to
IMG_SIZE0 = 97 # First dimension of image, should not be changed
IMG_SIZE1 = 90 # Second dimension of image, should not be changed


# Import images to DataFrame
df = pd.read_pickle(FROM_FILE) # Change to correct name of pickle file with images and data


# Constants
N_ROTATIONS = 30 # Number of rotations per image

df = df[df["Hand"]==1] # To only use one hand
df = df[df["Finger"]==0] # To only use one finger
df = df.sample(10) # To only use a few people

def crop_center(img):
    dim0, dim1 = img.shape
    x0 = (dim0-IMG_SIZE0)//2
    x1 = (dim1-IMG_SIZE1)//2
    if dim0 < IMG_SIZE0:
        cropped_img = np.ones((IMG_SIZE0,IMG_SIZE1))*255
        dim1_crop = img[:,x1:x1+IMG_SIZE1]
        cropped_img[:dim0,:] = dim1_crop # Put image to the left. Possible source of error
    else:
        cropped_img = img[x0:x0+IMG_SIZE0,x1:x1+IMG_SIZE1]
    return cropped_img


def rotate_image(img):
    # Constants
    shift0 = int(rnd.gauss(0,5)) # The choice of random function should perhaps be reconsidered
    shift1 = int(rnd.gauss(0,5))
    deg = rnd.gauss(0,45) # Uncomment for gaussian distribution
    # deg = rnd.uniform(0,360) # Uncomment for uniform distribution
    # Necessary modification
    img = np.reshape(img, (IMG_SIZE0,IMG_SIZE1)) # If the input array is flattened
    # Transform the image
    new_img = ndimage.rotate(img, deg, reshape=True, cval=255) # cval gives value to fill added edges with
    new_img = ndimage.shift(new_img, [shift0, shift1], cval=255)
    new_img = crop_center(new_img)
    # plt.imshow(new_img, cmap="gray") # Uncomment to see image
    new_img = new_img.flatten() # Comment out if array should not be flattened
    return new_img


# Create list of rotated images and samples information
rot_images = []
for i in range(len(df)):
    data = df.iloc[i]
    img = data[0]
    for j in range(N_ROTATIONS):
        img_rot = rotate_image(img)
        rot_images.append([img_rot, data[1],data[2],data[3],data[4]])
# Store as DataFrame
new_df = pd.DataFrame(rot_images, columns=["Image","Identity","Gender","Hand","Finger"])


plt.imshow(np.reshape(list(new_df["Image"][30]), (IMG_SIZE0,IMG_SIZE1)), cmap="gray")



np.array(new_df["Image"])


## Save data as pickle
#new_df.to_pickle(TO_FILE)