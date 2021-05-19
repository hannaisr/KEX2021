import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix



str_path = "C:/Users/glas4/Pictures/archive/SOCOFing/mf"
data_dir = Path(str_path)


batch_size = 32
img_height = 50
img_width = 50



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  labels = "inferred",
  label_mode = "binary",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds= tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  labels = "inferred",
  label_mode = "binary",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


  
class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
"""
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
"""
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

# Predict class for X.
y_predicted = model.predict(val_ds)
# Confusion matrix
cm = confusion_matrix(val_ds[1], y_predicted, normalize='true') # Remove 'normalize="all"' to get absolute numbers
plt.figure()
sn.heatmap(cm, annot=True, cmap='RdPu')
plt.title('Confusion matrix for prediction of ')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

"""
y_pred=model.predict_classes(val_ds)

print(y_pred)

labels =  []
for x, y in val_ds:
  labels.append([labels, np.argmax(model.predict(x), axis=-1)])
"""