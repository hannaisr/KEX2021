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
import numpy as np



str_path = "C:/Users/glas4/Pictures/archive/SOCOFing/handness2"
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


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  labels = "inferred",
  label_mode = "binary",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print(val_ds)
