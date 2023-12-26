
## Dataset

The notebook starts by mounting Google Drive, uploading the Kaggle API key (kaggle.json), and downloading the Intel Image Classification dataset using Kaggle.

```bash
from google.colab import drive
drive.mount('/content/drive')

from google.colab import files

uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the Intel Image Classification dataset
!kaggle datasets download -d puneet6060/intel-image-classification

!unzip -q intel-image-classification.zip -d ./intel-image-classification
```
Link to the original problem statement on kaggle : https://www.kaggle.com/puneet6060/intel-image-classification

Files/directories used in the code:

categories :- jason file containig the 6 different classes ('buildings','forest','glacier','mountain','sea', 'street').

seg_train and seg_pred which are contained in a folder called classification which holds training and validation images.

seg_pred :- containing the images to be predicted.


## Libraries and Setup

The necessary libraries, including TensorFlow, are imported. The code sets up essential parameters such as image size, batch size, and class labels.

```bash
# Importing necessary libraries
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
# Ignoring warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
# Importing scikit-learn metrics for model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
# Importing OS module for file and directory operations
import os
from os import walk
# Importing TensorFlow for deep learning tasks
import tensorflow as tf
tf.random.set_seed(0)
tf.keras.backend.clear_session()
```


## Exploratory Data Analysis

The notebook includes functions to explore and visualize the dataset, check for class imbalances, and plot sampled images from the training and test sets.
## Data Preprocessing

The code utilizes TensorFlow's image_dataset_from_directory to load and preprocess training and testing images.

## Model Building
Two models are implemented: a feature extraction model using EfficientNetB0 and a fine-tuned model. The training history, loss curves, and fine-tuning results are visualized.
## Model Evaluation

The notebook evaluates the final model on the test dataset and provides a confusion matrix for performance analysis.
## Prediction on Unseen Data
![image](https://github.com/areebanaz941/Natural-Scene-Classification-using-CNN/assets/129813908/02e7eb58-9c47-4750-b94f-162f9027ff58)

A few images from the prediction set are randomly selected, and the model's predictions are visualized.
## Conclusion

The notebook concludes with an overview of the implemented tasks and provides insights into the model's performance.
## Graphs and Results

Graphs and results, including loss curves, training and validation accuracy, and confusion matrix, are visualized within the notebook.
![image](https://github.com/areebanaz941/Natural-Scene-Classification-using-CNN/assets/129813908/e25091e7-3190-4fab-a91b-f6be2e179d73)

![image](https://github.com/areebanaz941/Natural-Scene-Classification-using-CNN/assets/129813908/7e6c26a9-77c2-46bc-963e-c36539e51c2a)

![image](https://github.com/areebanaz941/Natural-Scene-Classification-using-CNN/assets/129813908/f297073b-a042-48fc-98c5-15ecd9ee70d7)

