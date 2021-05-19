import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import layers
from keras.models import Model
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
import os
import random
from sklearn.metrics import confusion_matrix
#from google.colab import drive
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

img_size = 90

#######################################################################################################3333
###########################################################################################################3333

#df = pd.read_pickle('/content/drive/MyDrive/coolab/org_imgs_newcrop_3d.pkl')
#df = pd.read_pickle('/content/drive/MyDrive/coolab/org_imgs_newcrop_3d.pkl')
#df = pd.read_pickle('/content/drive/MyDrive/coolab/10_ppl_1000_rot.pkl')


df  = pd.read_pickle("/Users/glas4/Desktop/dnnfolder/org_imgs_newcrop_REAL.pkl")
df1 = pd.read_pickle("/Users/glas4/Desktop/dnnfolder/org_imgs_newcrop_EASY.pkl")
df2 = pd.read_pickle("/Users/glas4/Desktop/dnnfolder/org_imgs_newcrop_MEDIUM.pkl")
df3 = pd.read_pickle("/Users/glas4/Desktop/dnnfolder/org_imgs_newcrop_HARD.pkl")

"""
df = pd.read_pickle('/content/drive/MyDrive/coolab/org_imgs_newcrop_REAL.pkl')
df1 = pd.read_pickle('/content/drive/MyDrive/coolab/org_imgs_newcrop_EASY.pkl')
df2 = pd.read_pickle('/content/drive/MyDrive/coolab/org_imgs_newcrop_MEDIUM.pkl')
df3 = pd.read_pickle('/content/drive/MyDrive/coolab/org_imgs_newcrop_HARD.pkl')
"""

print(df.iloc[1]) # Just to check that everything looks fine

# Choose which columns to be data (X) and target (y)
x_name = "Image" # The data to be categorized, should be "Image"
y_label = "Gender" # The target label. In the end, Identity
#REAL
x_real = list(df[x_name])
y_real = df[y_label]
#EASY
x_easy = list(df1[x_name])
y_easy = df1[y_label]
#MEDIUM
x_medium = list(df2[x_name])
y_medium = df2[y_label]
#HARD
x_hard = list(df3[x_name])
y_hard = df3[y_label]

X, y = [], []
x_data = np.concatenate([x_easy, x_medium, x_hard], axis=0)
y_data = np.concatenate([y_easy, y_medium, y_hard], axis=0)

X = np.array(x_data).reshape(-1, img_size, img_size, 1)
#X = X / 255.0

y = np.array(y_data)
#print(np.shape(x_data)) # Should be ([number of images], [number of pixels])

# Divide into training and test sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)




#################   MAKE THE SHAPE (X,X,X,1) ###################


x_test = np.array(x_real).reshape(-1, img_size, img_size, 1)
x_test = x_test / 255.0
y_test = np.array(y_real)




img =  []
for label in x_data:
    img.append(label)
x_in_data = np.array(img).reshape(-1, img_size, img_size, 1)
x_data = x_in_data / 255.0

img3 =  []
for label in x_train:
    img3.append(label)
train_data = np.array(img3).reshape(-1, img_size, img_size, 1)
x_train = train_data / 255.0

img2 =  []
for label in x_val:
    img2.append(label)
val_data = np.array(img2).reshape(-1, img_size, img_size, 1)
x_val = val_data / 255.0

#################################################################
print("SHAPES;")
print(x_data.shape, y_data.shape)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
"""
###################################    CREATES A BLUR AND SCALE OF IMAGES
augs = [x_data[40000]] * 9

seq = iaa.Sequential([
    # blur images with a sigma of 0 to 0.5
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.Affine(
        # scale images to 90-110% of their size, individually per axis
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        # translate by -10 to +10 percent (per axis)
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        # rotate by -30 to +30 degrees
        rotate=(-30, 30),
        # use nearest neighbour or bilinear interpolation (fast)
        order=[0, 1],
        # if mode is constant, use a cval between 0 and 255
        cval=255
    )
], random_order=True)

augs = seq.augment_images(augs)

plt.figure(figsize=(16, 6))
plt.subplot(2, 5, 1)
plt.title('original')
plt.imshow(x_data[40000].squeeze(), cmap='gray')
for i, aug in enumerate(augs):
    plt.subplot(2, 5, i+2)
    plt.title('aug %02d' % int(i+1))
    plt.imshow(aug.squeeze(), cmap='gray')
"""
"""
##########################################################3
x1 = layers.Input(shape=(90, 90, 1))
x2 = layers.Input(shape=(90, 90, 1))
# share weights both inputs
inputs = layers.Input(shape=(90, 90, 1))
feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
feature = layers.MaxPooling2D(pool_size=2)(feature)
feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(feature)
feature = layers.MaxPooling2D(pool_size=2)(feature)
feature_model = Model(inputs=inputs, outputs=feature)

# 2 feature models that sharing weights
x1_net = feature_model(x1)
x2_net = feature_model(x2)
# subtract features
net = layers.Subtract()([x1_net, x2_net])
net = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(net)
net = layers.MaxPooling2D(pool_size=2)(net)
net = layers.Flatten()(net)
net = layers.Dense(64, activation='relu')(net)
net = layers.Dense(1, activation='sigmoid')(net)

model = Model(inputs=[x1, x2], outputs=net)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
################################################################################
"""
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (90,90,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))
model.summary()
################################################################################


epochs = 1#30 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
model_path = './Model.h5'


model.compile(optimizer = 'adam' , loss = "binary_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1),
    ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
]


history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (x_val, y_val), verbose = 1, callbacks= callbacks)
model.save('model.h5')
# Predict class for X.
y_predicted = model.predict(x_test)

print(y_predicted)
print(y_test)
print(y_test[0])
print(y_predicted[0])
print(np.shape(y_predicted))
print(np.shape(y_test))
# Confusion matrix
cm = confusion_matrix(y_test, y_predicted, normalize='true') # Remove 'normalize="all"' to get absolute numbers
plt.figure()
sn.heatmap(cm, annot=True, cmap='RdPu')
plt.title('Confusion matrix for prediction of '+y_name.lower())
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()