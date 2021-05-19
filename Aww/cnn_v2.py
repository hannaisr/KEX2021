import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#Import necessary libraries
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf

df = pd.read_pickle('org_imgs_newcrop.pkl')

# Get data
#df = pd.read_pickle('10_thumbs_expanded_dataset.pkl')

# Choose which columns to be data (X) and target (y)
X_name = "Image" # The data to be categorized, should be "Image"
y_name = "Identity" # The target label. In the end, Identity
X = list(df[X_name])
y = df[y_name]

print(np.shape(X))



X = [x/255 for x in X] # Scale the data to be in range [0,1]

# Divide into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

X_train= np.array(X_train)
X_test= np.array(X_test)


sample_shape = np.shape(X_train)
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1,1)
sample_shape2 = np.shape(X_test)
img_width2, img_height2 = sample_shape2[0], sample_shape2[1]
input_shape2 = (img_width2, img_height2, 1,1)
# Reshape data 
X_train = X_train.reshape(len(X_train), input_shape[1], 1, 1)
X_test  = X_test.reshape(len(X_test), input_shape2[1], 1, 1)


model = Sequential([
                    Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_uniform', input_shape = [97, 90, 1]),
                    MaxPooling2D(2),
                    Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(128, kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(2, activation = 'softmax'),
                    ])

model.compile(optimizer = optimizers.Adam(1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train,  y_train,epochs = 10, callbacks = [early_stopping_cb], verbose = 1,validation_data=(X_test, y_test))

