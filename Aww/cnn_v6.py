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
from google.colab import drive


df = pd.read_pickle('/content/drive/MyDrive/coolab/org_imgs_newcrop.pkl')
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


model = Sequential([
                    Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_uniform', input_shape = [90, 90, 1]),
                    MaxPooling2D(2),
                    Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(128, kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(1, activation = 'softmax'),
                    ])



model.compile(optimizer = optimizers.Adam(1e-3),loss = 'categorical_crossentropy', metrics = ['accuracy'])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(X_train, y_train, batch_size = 128, epochs = 5, 
          validation_split = 0.2, callbacks = [early_stopping_cb], verbose = 1)
#model.fit(X_train, y_train, batch_size = 128, epochs = 10)



# Predict class for X.
y_predicted = model.predict(X_test)
print(y_predicted[0])
print(y_test[0])
"""
# Confusion matrix
cm = confusion_matrix(y_test, y_predicted, normalize='true') # Remove 'normalize="all"' to get absolute numbers
plt.figure()
sn.heatmap(cm, annot=True, cmap='RdPu')
plt.title('Confusion matrix for prediction of '+y_name.lower())
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
"""