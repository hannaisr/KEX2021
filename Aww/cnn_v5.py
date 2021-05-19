#import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
#Import necessary libraries
import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Dense, Flatten,UpSampling2D
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization

df = pd.read_pickle("org_imgs_newcrop.pkl")
print(np.shape(df))
print(len(df))

X_name = "Image" # The data to be categorized, should be "Image"
y_name = "Gender" # The target label. In the end, Identity
X = list(df[X_name])
y = df[y_name]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(len(X_test))
print(len(X_train))
img_size = 90
img =  []
for label in X_train:
    img.append(label)
train_data = np.array(img).reshape(-1, img_size, img_size, 1)
X_train = train_data / 255.0

img2 =  []
for label in X_test:
    img2.append(label)
test_data = np.array(img2).reshape(-1, img_size, img_size, 1)
X_test = test_data / 255.0



print(np.shape(X_train))


"""
print(np.shape(X_train))
print(X_train)
X_train = np.array(X_train)
print(X_train)
print(np.shape(X_train))
img_size = 90
X_train = np.array(X_train).reshape(-1, img_size, img_size, 1)
X_train = X_train / 255.0
print(X_train)
print(np.shape(X_train))
X_train = np.array(X_train)
print(np.shape(X_train))
"""

model = Sequential([
                    Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_uniform', input_shape = [90, 90, 1]),
                    MaxPooling2D(2),
                    Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(128, kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(2, activation = 'softmax'),
                    ])

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded
"""
batch_size = 128
epochs = 200
inChannel = 1
x, y = 90, 90
input_img = Input(shape = (x, y, inChannel))

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder_train = autoencoder.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, y_test))
"""


model.compile(optimizer = optimizers.Adam(1e-3),loss = 'categorical_crossentropy', metrics = ['accuracy'])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(X_train, y_train, batch_size = 128, epochs = 30, 
          validation_split = 0.2, callbacks = [early_stopping_cb], verbose = 1)
#model.fit(X_train, y_train, batch_size = 128, epochs = 10)

# Predict class for X.
y_predicted = model.predict(X_test)
print(y_predicted)
print(y_test)
# Confusion matrix
cm = confusion_matrix(y_test, y_predicted, normalize='true') # Remove 'normalize="all"' to get absolute numbers
plt.figure()
sn.heatmap(cm, annot=True, cmap='RdPu')
plt.title('Confusion matrix for prediction of '+y_name.lower())
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()