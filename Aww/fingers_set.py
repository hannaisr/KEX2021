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




str_path = "C:/Users/glas4/Pictures/archive/SOCOFing/mf"
data_dir = Path(str_path)


batch_size = 32
img_height = 90
img_width = 90

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  labels = "inferred",
  label_mode = "binary",
  seed=1,
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
"""
h_list = []
for element in val_ds:
  for i in element[1]:
    #print(i.numpy()[0])
    h_list.append(int(i.numpy()[0]))
"""
  
class_names = train_ds.class_names

print(class_names)

"""
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype(int))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()
"""

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(10./255)
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

num_classes = 2

model = tf.keras.Sequential([
  #layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 1, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 1, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 1, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 1, activation='relu'),
  layers.MaxPooling2D(),

  layers.Flatten(),
  #layers.Dense(128, activation='relu'),
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

"""
labels = ["M","F"]
predictions = ["1 0"]
congfu = tf.confusion_matrix(labels, predictions,num_classes)
print(congfu)
"""

y_pred=model.predict_classes(val_ds)
print(y_pred)
print(h_list)
h_list = numpy.argmax(test_y, axis=1)
"""
rr =0
for i in range(len(y_pred)):
  e = h_list[i]+y_pred[i]
  if e == 1:
    rr +=1
print(rr/len(y_pred))
"""

#con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
"""
predictions = np.array([])
labels =  np.array([])
for x, y in val_ds:
  predictions = np.concatenate([predictions, model.predict_classes(x)])
  labels = np.concatenate([labels, np.argmax(model.predict(x), axis=-1)])




"""
"""
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
print(con_mat_norm)
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = class_names, 
                     columns = classes_names)

figure = plt.figure(figsize=(2, 2))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
"""