# Histopathologic Cancer Detection

### Problem Statement
This project aims to develop a deep learning algorithm to identify metastatic cancer in small image patches taken from larger whole-slide images. The algorithm is trained on a set of labeled images, and evaluated on a test set.   Each training image patch is labeled 1 if it has tumor and 0 if not. Given an image patch, the algorithm predicts tumor probability.

### Datasets and Inputs
The dataset comes from a [Kaggle competition](https://www.kaggle.com/c/histopathologic-cancer-detection/data). It has 220,026 pathology images in the training set and 57,459 pathology images in the test set.  Each image in the training set is labeled with 1 or 0. The label 1 indicates that the center 32x32px region of an image contains at least one pixel of tumor tissue.

### Metrics
In this Kaggle competition, solution models are evaluated on area under the ROC curve between the predicted probability and the observed target on the test set. The ROC curve is a plot of True Positive Rate against False Positive Rate as the threshold probability dividing the positive and negative varies.   The area under the ROC curve (AUC-ROC) represents the performance of a solution model.  An excellent model has the area under the ROC curve close to 1. When the area under the ROC curve is 0.5, the model has no classification capability.  When the area under the ROC curve is close to 0, the model predicts the opposite of the ground truth.

### Exploratory Data Analysis
* See implementation in [exploratory_data_analysis](exploratory-data-analysis.ipynb) and result in [report](report.pdf)




### Software
* Python3
* Jupyter notebook

### Libraries
* numpy
* pandas
* matplotlib
* os
* cv2
* scipy
* sklearn
* random
* tqdm
* glob
* keras
* fastai
