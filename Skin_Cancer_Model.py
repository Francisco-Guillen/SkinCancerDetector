import pandas as pd
#math operations
import numpy as np
#machine learning
import cv2
import os
from random import shuffle
from tqdm import tqdm
import random
#for opening and loading image
from PIL import Image
#for preprocessing
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
#Doing One hot encoding as classifier has multiple classes
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from random import shuffle
#For augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#MobileNetV2 model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Model, layers
from numpy import loadtxt

import itertools
from sklearn.metrics import confusion_matrix,classification_report

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Data Preparation

data=[]
labels=[]
Basal=os.listdir("/Training_Input/Basal_Cell_Carcinoma/Train/")
for a in Basal:
    try:
        image=cv2.imread("/Training_Input/Basal_Cell_Carcinoma/Train/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((224, 224))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")

Melanoma=os.listdir("Training_Input/Melanoma/Train2/")
for b in Melanoma:
    try:
        image=cv2.imread("/Training_Input/Melanoma/Train2/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((224, 224))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")
Nevus=os.listdir("/Training_Input/Nevus/Train/")
for c in Nevus:
    try:
        image=cv2.imread("/Training_Input/Nevus/Train/"+c)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((224, 224))
        data.append(np.array(size_image))
        labels.append(2)
    except AttributeError:
        print("")

# Converting features and labels in array
feats=np.array(data)
labels=np.array(labels)
# Saving features and labels for later
np.save("/Training_Input/feats_train",feats)
np.save("/Training_Input/labels_train",labels)

# Loading Saved Data and Labels

feats=np.load("/Training_Input/feats_train.npy")
labels=np.load("/Training_Input/labels_train.npy")

# Randomizing Data and Labels

s=np.arange(feats.shape[0])
np.random.shuffle(s)
feats=feats[s]
labels=labels[s]
num_classes=len(np.unique(labels))
len_data=len(feats)
print(len_data)
print(num_classes)

# Train Test Split

(x_train,x_test)=feats[(int)(0.2*len_data):],feats[:(int)(0.2*len_data)]

(y_train,y_test)=labels[(int)(0.2*len_data):],labels[:(int)(0.2*len_data)]

# Normalization

x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by dividing 255.
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)

y_train=to_categorical(y_train,3)
y_test=to_categorical(y_test,3)

# Image Augmentation

trainAug  = ImageDataGenerator(
featurewise_center=False,  # Set input mean to 0 over the dataset
        samplewise_center=False,  # Set each sample mean to 0
        featurewise_std_normalization=False,  # Divide inputs by std of the dataset
        samplewise_std_normalization=False,  #Divide each input by its std
        zca_whitening=False,  # Apply ZCA whitening
        rotation_range=10,  # Randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # Randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # Randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # Randomly flip images
        vertical_flip=False)

# Model Building

conv_base = MobileNetV2(
    include_top=False,
    input_shape=(224, 224, 3),
    weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = True

x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.1)(x)
predictions = layers.Dense(3, activation='softmax')(x)
model = Model(conv_base.input, predictions)

callbacks = [ModelCheckpoint('.mdl_wts.hdf5', monitor='val_loss',mode='min',verbose=1, save_best_only=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='min', min_lr=0.00000000001)]


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
BS = 64
print("[INFO] training head...")
H = model.fit(
	trainAug.flow(x_train,y_train, batch_size=BS),
	steps_per_epoch=train_len // BS,
	validation_data=(x_test, y_test),
	validation_steps=test_len // BS,
	epochs=30,callbacks=callbacks)

accuracy = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])

model = load_model('.mdl_wts.hdf5')
model.save('/Models/model_v1.h5')
