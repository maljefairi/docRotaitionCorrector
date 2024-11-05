# Synthetic Document Generator

A Python tool for generating synthetic document images with text in multiple languages and training an orientation detection model using deep learning.

## Features

- Generates synthetic documents with text in 13 different languages
- Supports multiple document formats (licenses, passports, A4)
- Applies random rotations and perspective transformations
- Includes a deep learning model to detect document orientation
- Uses Spatial Transformer Networks (STN) for improved orientation detection

## Document Types & Sizes

- Driver's License (850x540 pixels)
- Passport (1250x875 pixels) 
- A4 Document (2480x3508 pixels)

## Supported Languages

The generator supports text in:
- English
- Chinese (简体中文)
- Arabic (العربية)
- Romanian
- Urdu (اردو)
- Greek (Ελληνικά)
- Spanish (Español)
- Italian (Italiano)
- Portuguese (Português)
- French (Français)
- Thai (ไทย)
- Korean (한국어)
- Russian (Русский)

## Technical Details

- Uses PyTorch for the deep learning model
- Implements ResNet18 with Spatial Transformer Networks
- Generates synthetic data with OpenCV and PIL
- Outputs sine/cosine predictions for angle regression
- Includes data augmentation with random transformations

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- PIL
- NumPy
- tqdm

## Installation

1. Clone this repository:
