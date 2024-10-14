# Aasmaan_modified

# SE-ResNet50 for Satellite Image Classification

This repository contains code for classifying high-resolution satellite images using SE-ResNet50 and other advanced deep learning models. The dataset used is the UCMerced LandUse dataset, and the code includes models like SE-ResNet50, SE-ResNeXt, and SENet, using transfer learning to fine-tune pretrained models for image classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Project Overview
This project aims to classify high-resolution satellite images from the UCMerced LandUse dataset into various land use categories. The key objectives are:
- Convert `.tif` images to `.jpg` format for easier processing.
- Use SE-ResNet50, SE-ResNeXt, and SENet models for high accuracy classification.
- Train models on the dataset and evaluate performance with metrics such as accuracy, confusion matrix, and ROC curves.

## Dataset
The dataset used for this project is the [UCMerced LandUse Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html). It contains 21 land-use classes with 100 images per class. Each image is 256x256 pixels with a spatial resolution of 0.3 meters.

- **Input:** `.tif` satellite images.
- **Output:** Classified land use images in categories like agricultural, residential, commercial, etc.

## Preprocessing
Before feeding the images into the model, we perform the following preprocessing steps:
1. Convert `.tif` images to `.jpg` format using the Python Imaging Library (PIL).
2. Resize images to 224x224 pixels.
3. Normalize images using ImageNet mean and standard deviation values.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Model Architectures
We have used the following models, all of which are based on the Squeeze-and-Excitation (SE) block:
- **SE-ResNet50**: A variant of ResNet50 with SE blocks.
- **SE-ResNeXt50/101**: A more advanced version of SE-ResNet with additional cardinality for wider networks.
- **SENet154**: The most complex and accurate model in the family, incorporating SE blocks.

## Training
- We trained the models on the UCMerced dataset using the following configuration:
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
  - Learning Rate: 0.001
  - Epochs: 35
  - Batch Size: 32

Example training loop:
```python
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
```

## Evaluation
We evaluated the model using:
- **Accuracy**: The percentage of correct predictions.
- **Confusion Matrix**: A heatmap to visualize misclassifications.
- **ROC Curve**: For multiclass classification, we plotted the ROC curve for each class.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
```

## Results
The models achieved the following accuracy on the validation set:
- **SE-ResNet50**: 94.5% accuracy
- **SE-ResNeXt50**: 95.3% accuracy
- **SENet154**: 96.2% accuracy

## Installation
To install the required dependencies, clone the repository and install the Python packages using `pip`:

```bash
git clone https://github.com/ArnavGhosh999/Aasmaan.git
cd Aasmaan/Se-ResNet50
pip install -r requirements.txt
```

Ensure that you have the following libraries installed:
- `torch`
- `torchvision`
- `timm`
- `Pillow`
- `rasterio`
- `opencv-python`

You can also install them manually using:
```bash
pip install torch torchvision timm Pillow rasterio opencv-python
```

## Usage
1. **Dataset Preparation**: Mount your Google Drive or local directory where the dataset is stored.
2. **Image Conversion**: Run the conversion script to convert `.tif` images to `.jpg`:
   ```python
   python convert_images.py
   ```
3. **Training**: Train the model using:
   ```python
   python train_model.py
   ```
4. **Evaluation**: Evaluate the model performance:
   ```python
   python evaluate_model.py
   ```

## References
- UCMerced LandUse Dataset: [Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
- SE-ResNet and SE-ResNeXt models: [Timm Library](https://github.com/rwightman/pytorch-image-models)
- SENet Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)


