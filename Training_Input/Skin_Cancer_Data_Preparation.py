import pandas as pd
df = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
df.head() #Each file belongs to a differente category


# Melanoma

melanoma_list = df[df['MEL']==1.0]['image'].tolist()
len(melanoma_list)

import os
import shutil
source = 'Training_Input/ISIC_2019_Training_Input/'
destination = 'Training_Input/Melanoma/Train2/'
allfiles = os.listdir(source)

for f in allfiles:
  
  m = f.split('.')[0]
  if m in melanoma_list:
    file = "melanoma_" + str (f)
    print(file)
    shutil.copy(source + f, destination + file)


# Basal Cell Carcinoma
    
BCC_list = df[df['BCC']==1.0]['image'].tolist()
len(BCC_list)

import os
import shutil
source = 'Training_Input/ISIC_2019_Training_Input/'
destination = 'Training_Input/Basal_Cell_Carcinoma/Train/'
allfiles = os.listdir(source)

for f in allfiles:
  
  m = f.split('.')[0]
  if m in BCC_list:
    file = "BCC_" + str (f)
    print(file)
    shutil.copy(source + f, destination + file)


# Nevus
    
Nevus_list = df[df['NV']==1.0]['image'].tolist()
len(Nevus_list)

import os
import shutil
source = 'Training_Input/ISIC_2019_Training_Input/'
destination = 'Training_Input/Nevus/Train/'
allfiles = os.listdir(source)

for f in allfiles:
  
  m = f.split('.')[0]
  if m in Nevus_list:
    file = "Nevus_" + str (f)
    print(file)
    shutil.copy(source + f, destination + file)
    
