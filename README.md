## Gradio Demo available!

You can check out a Gradio demo of the classifier on the HuggingFace Space linked here: [apziva-monreader-classifier](https://robpetrosino-apziva-monreader-demo.hf.space).

## Introduction

This repository contains the deep learning model that is trained on the set of images provided by Apziva for the `MonReader` project. The client is a leader in innovative computer vision solutions, and is seeking help to develop MonReader, a new mobile document digitization experience for the blind, researchers, and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. MonReader is a mobile app that detects page flips from low-resolution camera preview, takes a high-resolution picture of the document, recognizing its corners, and crops it accordingly; it dewarps the cropped document to obtain a bird's eye view, and sharpens the contrast between the text and the background; and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.

The dataset consists of frames extracted from page flipping videos from smart phones, which were eventually labelled as flipping and not flipping. The frame names are stored in the `data/raw` folder in sequential order with the following naming structure: `VideoID_FrameNumber`.

The goal of the project is to build a deep learning model that is able to predict if the page is being flipped using a single image.

## Prerequisites

- `numpy`, `os`
- `matplotlib`, `seaborn`
- `tensoryflow`
- `sklearn`

## Train and predict

Among the 5 different CNN models built via transfer learning, the one based on the `MobileNetV2` model showed an accuracy and F1 scores of about 95%, while still maintaining a size suitable for mobile application (~12 MB). The code for the model can be found in the `/models/train.py` file. The model was trained on the images stored in the folder `data/raw/images/training/`.

The code for using the trained model to make predictions on new data is contained in the `/models/test.py` file.

## Evaluation

The model uses accuracy and `f1_score` as evaluation metrics. The size of the model is also taken into account.

## Conclusion

In this project, I leveraged the power of a series of CNN models to help the mobile app Monreader to classify images showing and not showing page flips. All models (custom CNN, VGG16, ResNet, MobileNet, and EfficientNet) performed very well, but only the MobileMobile model had a suitable size, while keeping high accuracy scores. 
