This repository contains the deep learning model that is trained on the set of images provided by Apziva for the `MonReader` project. The client is a leader in innovative computer vision solutions, and is seeking help to develop MonReader, a new mobile document digitization experience for the blind, researchers, and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. MonReader is a mobile app that detects page flips from low-resolution camera preview, takes a high-resolution picture of the document, recognizing its corners, and crops it accordingly; it dewarps the cropped document to obtain a bird's eye view, and sharpens the contrast between the text and the background; and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.

The dataset consists of frames extracted from page flipping videos from smart phones, which were eventually labelled as flipping and not flipping. The frame names are stored in the `data/raw` folder in sequential order with the following naming structure: `VideoID_FrameNumber`.

The goal of the project is to build a deep learning model that is able to predict if the page is being flipped using a single image.

## Prerequisites

TBD

## Train and predict

TBD

## Evaluation

The model uses the `f1_score` as evaluation metrics.

## Conclusion

TBD