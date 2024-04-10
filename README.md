# Skin Cancer Detection Web App

This repository contains the source code of a web application for skin cancer detection using machine learning. The application is capable of receiving images of skin lesions and classifying them into three categories: Basal Cell Carcinoma (Cancer), Melanoma (Cancer), and Nevus (Non-Cancerous).

<p align="center">
<img src="https://github.com/Francisco-Guillen/SkinCancerDetector/assets/83434031/c6ed8d7f-f3d8-430f-8908-8f4ee3abecc9" width="900">
<br>
  Figure 1: Web Application Example
</p>

## Features

- Image Upload: Users can upload images of skin lesions for analysis.
- Cancer Detection: The application uses a machine learning model based on MobileNetV2 architecture to predict the probability of each class for the uploaded image.
- Generation of Heatmaps (Grad-CAM): Heatmaps are generated to visualize the regions of the image that contribute most to the classification decision of the model.
  
## Prerequisites
-  Python
- Opencv
- Tqdm
- Pillow
- Matplotlib
- Scikit-learn
- Seaborn
 -Pandas
- Numpy
- TensorFlow
- Flask
- PIL
- Matplotlib
- Werkzeug

## How to Run
1. Clone this repository.
2. Install dependencies: pip install -r requirements.txt
3. Download the dataset from [ISIC Archive](https://challenge.isic-archive.com/data/#2019).
4. Train the model by running: Skin_Cancer_Model.py
5. Run the application: app.py
6. Access the application in your browser at [http://localhost:8083](http://localhost:8083).

## Project Structure

- `app.py`: The main file defining the Flask application and its routes.
- `heatmap.py`: Contains functions for generating heatmaps using Grad-CAM.
- `models/model_v1.h5`: Pre-trained machine learning model (based on MobileNetV2 architecture) for skin cancer detection.
- `uploads/`: Directory to store images uploaded by users.
- `heatmap/`: Directory to store generated heatmaps.
- `templates/`: Directory containing HTML files used to render web pages.
- `static/`: Directory containing CSS and JavaScript files used to style HTML pages.

## Model Evaluation
<p align="center">
<img src="https://github.com/Francisco-Guillen/SkinCancerDetector/assets/83434031/de8bd139-c2e3-45d4-84c4-61377117380c" width="600">
<br>
  Figure 2: ROC Curves and AUC
</p>

## Refferences
- [Skin Cancer Detection Using Transfer Learning Deep CNN Approach](https://www.youtube.com/watch?v=t43VdRgWH98&t=139s)
- **BCN_20000 Dataset**: (c) Department of Dermatology, Hospital Clínic de Barcelona
- **HAM10000 Dataset**: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; [DOI: 10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
- **MSK Dataset**: (c) Anonymous; [arXiv:1710.05006](https://arxiv.org/abs/1710.05006), [arXiv:1902.03368](https://arxiv.org/abs/1902.03368) 

