import os 
import cv2
import numpy as np 
import pandas as pd 
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
from os.path import isfile, join

# For CNN 
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical, plot_model
from keras import layers, regularizers, optimizers, callbacks
from sklearn.model_selection import train_test_split 
from collections import defaultdict

# For Autoencoder
import keras
import gzip
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization



# setting paths
path = "/Users/glas4/Pictures/archive/SOCOFing"
real_dir = '/Users/glas4/Pictures/archive/SOCOFing/Real'
altered_dir = "/Users/glas4/Pictures/archive/SOCOFing/Altered"
# Forming a dataframe to 
data = []
df=pd.DataFrame(data,columns=['img_id','gender', 'hand', 'finger', 'alteration'])

d = defaultdict(lambda: "Not Present")
d["Right"] = 1
d["Left"] = 0
d["M"] = 0
d["F"] = 1
d["thumb"] = 0
d["index"] = 1
d["middle"] = 2
d["ring"] = 3
d["little"] = 4
d["No"] = 0
d["Obl"] = 1
d["CR"] = 2
d["Zcut"] = 3

for dirname, dirnames, filenames in os.walk(path ):
    print(dirname + ":")
    alteration='No'
    for filename in filenames:#print(filename)
        #if isfile(filename)
        img, ext = os.path.splitext(filename)
        img_id, name = img.split('__')
        if(dirname != real_dir):
            gender, hand, finger, name, alteration = name.split('_')
        else:
            #print("ss")
            gender, hand, finger, name = name.split('_')
        a0 = int(img_id)
        a1 = d[gender]
        a2 = d[hand]
        a3 = d[finger]
        a4 = d[alteration]
        data = [[a0, a1, a2, a3, a4]]
        df1=pd.DataFrame(data,columns=['img_id','gender', 'hand', 'finger', 'alteration'])
        df = df.append(df1)
    print("image count: \t",len(filenames), "\n")

print("dataframe shape: \t",df.shape, "\n\n")
print(df.dtypes)

# Histogram 
nHistogramShown = 5
nHistogramPerRow = 5

nunique = df.nunique()
df = df[[col for col in df[0:4]]]
nRow, nCol = df.shape
columnNames = list(df)
nHistRow = (nCol + nHistogramPerRow - 1) / nHistogramPerRow
plt.figure(num=None, figsize=(6*nHistogramPerRow, 8*nHistRow), dpi=80, facecolor='w', edgecolor='k')
for i in range(min(nCol, nHistogramShown)):
    plt.subplot(nHistRow, nHistogramPerRow, i+1)
    df.iloc[:,i].hist()
    plt.ylabel('counts')
    plt.xticks(rotation=90)
    plt.title(f'{columnNames[i]} (column {i})')
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show()

# Ploting different types of images from the dataset provided
real = matplotlib.image.imread(join(real_dir, "1__M_Left_index_finger.BMP"))
alt_cr = matplotlib.image.imread(join(altered_dir, "Altered-Hard", "1__M_Left_index_finger_CR.BMP"))
alt_obl = matplotlib.image.imread(join(altered_dir, "Altered-Hard", "1__M_Left_index_finger_Obl.BMP"))
alt_zcut = matplotlib.image.imread(join(altered_dir, "Altered-Hard", "1__M_Left_index_finger_Zcut.BMP"))

plt.figure()
plt.figure(figsize=(24, 25))
plt.subplot(1, 5, 1, facecolor='w')
plt.imshow(real, cmap='gray')
plt.subplot(1, 5, 2, facecolor='w')
plt.imshow(alt_cr, cmap='gray')
plt.subplot(1, 5, 3, facecolor='w')
plt.imshow(alt_obl, cmap='gray')
plt.subplot(1, 5, 4, facecolor='w')
plt.imshow(alt_zcut, cmap='gray')


def load_images(path, train=True):# Data loading function 
    print("Loading data from: ", path)
    data = []
    for img in os.listdir(path):
        name, ext = os.path.splitext(img)
        ID, etc = name.split('__')
        ID = int(ID) - 1
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img_array, (96, 96))
        data.append([ID, img_resize])
    return data


# importing images data using loading function
Easy_images = load_images(altered_dir+"/Altered-Easy", train=True)
Medium_images = load_images(altered_dir+"/Altered-Medium", train=True)
Hard_images = load_images(altered_dir+"/Altered-Hard", train=True)
Real_images = load_images(real_dir, train=False)
#merging altered data
Altered_images = np.concatenate([Easy_images, Medium_images, Hard_images], axis=0)

del Easy_images, Medium_images, Hard_images


# Data extracting and preprocessing
x_altd, y_ID_altd = [], []

for ID, img in Altered_images:
    x_altd.append(img)
    y_ID_altd.append(ID)    

x_altd = np.array(x_altd).reshape(-1, 96, 96, 1)
x_altd = x_altd / 255.0
y_ID_altd = to_categorical(y_ID_altd, num_classes=600) # 600 persons in total


# Data spliting 
Imgs_train, Imgs_val, ID_train, ID_val = train_test_split(
    x_altd, y_ID_altd, test_size=0.2, random_state=2)
del x_altd, y_ID_altd

# Setting up Convolutional Neural Network Architecture 
model = [0]
model_name = 'Fingerprint_Model'
model = Sequential(name= model_name)

model.add(layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape = (96, 96, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128,(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(600, activation='softmax'))

# Complete with Adam optimizer and entropy cost
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Hyperparameters 
history = [0] 
ReduceLR_minlr = 1e-9
epochs = 15
batch_size = 128
CallBack = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=7, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.1, patience=0.9, min_lr=ReduceLR_minlr, verbose=1),
    callbacks.TensorBoard(log_dir="./log_dir/"+model_name)]

# Training the fingerprint model 
history = model.fit(Imgs_train, ID_train,
                    batch_size = batch_size,
                    epochs = epochs, 
                    validation_data = (Imgs_val, ID_val),
                    verbose = 1, callbacks= CallBack)
# Saving fingerprint model
model.save('model.saved')

del Imgs_train, Imgs_val, ID_train, ID_val, Altered_images

# Ploting accuracy and loss values
pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)

# Loading saved fingerprint model 
model = load_model('model.saved')
count = 0

# Prediction and evaluation trained model

for ID, feature in Real_images:
    img = np.array(feature).reshape( -1, 96, 96, 1)
    pred_id = model.predict_classes(img)
    error = ID-pred_id
    if error != 0:
        #print("wrong prediction", pred_id)
        count = count +1
        
# Count of wrongly predicted images
print("Error count : ", count)
# Percentage error for real images
print("Percentage error : ",count/60, " % ")

del Real_images















