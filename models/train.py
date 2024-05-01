# basics
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# neural networks
from tensorflow.keras import layers
from tensorflow.keras import backend as bk
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# pre-trained models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 # MobileNet (version 2)

# load data and batch
train_data_dir = '../data/raw/images/training'
img_height, img_width, img_chn = 200, 200, 3
batch_size = 16

train_imgs = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.15,
  subset='training',
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size,
)

val_imgs= tf.keras.preprocessing.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.15,
  subset='validation',
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

classes = train_imgs.class_names

## transfer modeling
 def mobilenet_transfer(train_imgs, val_imgs, img_width, img_height, img_chn, classes):
    # load the model
    mobilenet = MobileNetV2(
      weights='imagenet', # fetching the pretrained hyperparameters
      input_shape = (img_width, img_height, img_chn),
      classes = classes,
      include_top=False,
      pooling='avg'
    )

    # freeze the weights of the remaining layers so they are not retrained during the training process
    for layer in mobilenet.layers:
      layer.trainable=False

    last_layer = mobilenet.layers[-1].output # remove the last layer of the pre-trained model
    last_layer = layers.Flatten()(last_layer) # flatten it
    last_layer = layers.Dense(256, activation='relu')(last_layer) # add a new hidden layer
    last_layer = layers.Dropout(0.5)(last_layer) # to avoid overfitting
    output_layer = layers.Dense(classes, activation='softmax')(last_layer)

    mobilenet_model = Model(inputs=mobilenet.input, outputs=output_layer)
    mobilenet_model.compile(
      optimizer = 'adam',
      loss = 'sparse_categorical_crossentropy',
      )

    trained_mobilenet = mobilenet_model.fit(
       train_imgs,
       epochs=10,
       validation_data=val_imgs
    )

    return mobilenet_model, trained_mobilenet

mobilenet_model, trained_mobilenet = mobilenet_transfer(train_imgs, val_imgs, img_width, img_height, img_chn, len(classes))

save_dir = './models/'
mobilenet_save_dir = save_dir + 'mobilenet.keras'
mobilenet_model.save(mobilenet_save_dir)
