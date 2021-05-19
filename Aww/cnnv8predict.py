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


from tensorflow import keras
model = keras.models.load_model('model.h5')

# Predict class for X.
y_predicted = model.predict(x_test)
yy=[]
for ii in y_predicted:
	if ii >0.5:
		yy.append(1)
	else:
		yy.append(0)

print(yy[100:120])
print(y_test[100:120])

# Confusion matrix
cm = confusion_matrix(y_test, yy) # Remove 'normalize="all"' to get absolute numbers
plt.figure()
sn.heatmap(cm, annot=True, cmap='RdPu')
plt.title('Confusion matrix for prediction of '+y_label.lower())
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


#from keras.utils.np_utils import to_categorical
#y_test  = to_categorical(y_test, num_classes = 2)
#print(y_test)
#print(x_test[0])
model.evaluate(x_test, y_test)

