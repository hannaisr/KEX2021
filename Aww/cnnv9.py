# --------------------------------------
# IMPORTS
import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Sequential 
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten,Dropout, BatchNormalization


# --------------------------------------
# CONSTANTS & PATHS
img_size = 96
REAL = "/Users/glas4/Pictures/archive/SOCOFing/Real"
EASY_ALTERED = "/Users/glas4/Pictures/archive/SOCOFing/Altered/Altered-Easy"
MEDIUM_ALTERED = "/Users/glas4/Pictures/archive/SOCOFing/Altered/Altered-Medium"
HARD_ALTERED = "/Users/glas4/Pictures/archive/SOCOFing/Altered/Altered-Hard"


# --------------------------------------
# OBTAINS LABELS FROM FINGERPRINT FILES
def get_label(img_path,train = True):
    filename, _ = os.path.splitext(os.path.basename(img_path))
    store_path, labelss = filename.split('__')
    if train:
        gender, hand, finger, _, _ = labelss.split('_')
    else:
        gender, hand, finger, _ = labelss.split('_')
    
    #Gender to int
    if gender == 'M':
        gender = 0
    else:
        gender = 1

    #Hand to int
    if hand == 'Left':
        hand = 0
    else: hand = 1
  
    #Finger to int
    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4

    return np.array([gender], dtype=np.uint16)


# --------------------------------------
# GENERATE THE DATASET FROM IMAGES
def load(path,prep):
    dataset_list = []
    for img in os.listdir(path):
            img_list = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) #Reads image
            img_crop = cv2.resize(img_list, (img_size, img_size))                #Crops image
            label = get_label(os.path.join(path, img),prep)                     #Gets label for image
            dataset_list.append([label[0], img_crop])                            #Adds to dataset
    return dataset_list

# --------------------------------------
# CREATES ALL DATASETS
Easy_list = load(EASY_ALTERED, prep = True)
print("1")
Medium_list = load(MEDIUM_ALTERED, prep = True)
print("2")
Hard_list = load(HARD_ALTERED, prep = True)
print("3")
test = load(REAL, prep = False)
data = np.concatenate([Easy_list, Medium_list, Hard_list], axis=0)
#random.shuffle(test)
#random.shuffle(data)


# --------------------------------------
# CREATES THE TRAINING SET
labels, img = [], []
for label, feature in data:
    labels.append(label)
    img.append(feature)
dataset_train = np.array(img).reshape(-1, img_size, img_size, 1)
dataset_train = dataset_train / 255.0
dataset_train_labels = to_categorical(labels, num_classes = 2)

# --------------------------------------
# SETUP FOR MODEL
model = Sequential([
                    Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_uniform', input_shape = [96, 96, 1]),
                    MaxPooling2D(2),
                    Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(128, kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(2, activation = 'softmax'),
                    ])

#Conv2D        |    
#MaxPooling2D  |    
#Flatten       |    
#Dense         |    

#input_shape = [96, 96, 1]        | Image size is 96x96 pixels
#kernel_initializer='he_uniform'  |   
#activation = 'softmax'           |
#activation='relu'                |







# --------------------------------------
# CREATES MODEL OR LOAD MODEL
model.compile(optimizer = optimizers.Adam(1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dataset_train, dataset_train_labels, batch_size = 128, epochs = 4, validation_split = 0.2, verbose = 1)#, callbacks = [early_stopping_cb])
model.save('model.h5')
model.summary()

# --------------------------------------
# LOAD MODEL
#model = keras.models.load_model('model.h5')
#print("LOADING, PLEASE WAIT...")


# --------------------------------------
# EVALUATION OF MODEL
test_images, test_labels = [], []
for label, feature in test:
    test_images.append(feature)
    test_labels.append(label)
test_images = np.array(test_images).reshape(-1, img_size, img_size, 1)
test_images = test_images / 255.0
test_labels = to_categorical(test_labels, num_classes = 2)
model.evaluate(test_images, test_labels)
y_predicted = model.predict(test_images)
yy = []
#y_list = model.predict(tests)
#pd.DataFrame(history.history).plot(figsize = (8,5))
#plt.grid()
#plt.x_label("Tests")
#plt.gca().set_ylim(0,1)
y_real_label = []
for i in range(len(y_predicted)):
	if y_predicted[i][0] > 0.5:
		yy.append(1)
	else:
		yy.append(0)
	y_real_label.append(int(test_labels[i][0]))
print(y_predicted[10:40])
print(test_labels[10:40])
print(yy[10:40])
print(y_real_label[10:40])
# Confusion matrix
cm = confusion_matrix(y_real_label, yy) # Remove 'normalize="all"' to get absolute numbers
plt.figure()
sn.heatmap(cm, annot=True, cmap='RdPu')
plt.title('Confusion matrix for prediction of gender')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
print("HE")
