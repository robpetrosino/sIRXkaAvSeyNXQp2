# basics
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# pre-trained models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 # MobileNet (version 2)

# metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

img_height, img_width, img_chn = 200, 200, 3
batch_size = 16

# load test set and batch
test_imgs = tf.keras.preprocessing.image_dataset_from_directory(
  test_data_dir,
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# load model
mobilenet_model = tf.keras.models.load_model('./models/mobilenet.keras', compile=False)

# evaluate model on the test set
def evaluate_model(model, test_data):
  eval_metrics = {}

  predicted_labels = []
  true_labels = []

  for images, labels in test_data:
    true_labels.extend(labels.numpy())
    predicted_labels.extend(tf.argmax(model.predict(images), axis=1).numpy())

  eval_metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
  eval_metrics['f1_score'] = f1_score(true_labels, predicted_labels)
  eval_metrics = pd.Series(eval_metrics)
  confusion_m = confusion_matrix(true_labels, predicted_labels)

  plt.figure(figsize=(10,10))
  sns.heatmap(confusion_m, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
  plt.xlabel('predicted labels')
  plt.ylabel('actual labels')
  plt.title('confusion matrix')
  plt.show()

  print(eval_metrics)
  return eval_metrics

mobilenet_evaluation = evaluate_model(mobilenet_model, test_imgs)
