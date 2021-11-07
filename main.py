import os
from os import walk
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

#Import Py Files
import directory

OS_path = directory.path('Mac Flask')

##CONTROLS
GENERATE_IMGS = True
CLASSIFY = True

#Load Model
load_mod = "MODEL/mod-04-10-2021-21-20-10-acc-78.246.pth"

#Training Parameters
BATCH_SIZE = 200
EPOCHS = 20

##Generate Images to Classifiy
if GENERATE_IMGS:
    exec(open("generate_images_classify.py").read())
    print('\n')

##CUDA INIT
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\nNET: Running on the GPU\n")
else:
    device = torch.device("cpu")
    print("\nNET: Running on the CPU\n")

HEIGHT_MULTIPLIER = 2

class Classify():

    RESIZE_WIDTH = 99
    RESIZE_HEIGHT = 13*HEIGHT_MULTIPLIER

    classify_data = []
    test_audio = "Images Classify"

    def make_classify_data(self):
        i = 0
        for f in tqdm(os.listdir(self.test_audio)):
                    try:
                        path = os.path.join(self.test_audio, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
                        self.classify_data.append([np.array(img)])
                        i += 1
                    except Exception as e:
                        pass
        np.save("classify_data.npy", self.classify_data)



training_data = np.load("training_data.npy", allow_pickle=True)

##CNN
class Net_shape(nn.Module):
     def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(1, 32, 3)
          self.conv2 = nn.Conv2d(32, 64, 3)
          self.conv3 = nn.Conv2d(64, 128, 3)
          self.pool1 = nn.MaxPool2d((2,2))
          self.pool2 = nn.MaxPool2d((2,2))
          self.pool3 = nn.MaxPool2d((2,2))

     def forward(self, x):
          x = F.relu(self.conv1(x))
          x = self.pool1(x)
          x = F.relu(self.conv2(x))
          x = self.pool2(x)
          x = F.relu(self.conv3(x))
          x = self.pool3(x)
          x = x.flatten(start_dim=1)

          return x.shape[1]

net_shape = Net_shape()
flatten_val = net_shape.forward(torch.randn(1,1,13*HEIGHT_MULTIPLIER,99))

class Net(nn.Module):
     def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(1, 32, 3)
          self.conv2 = nn.Conv2d(32, 64, 3)
          self.conv3 = nn.Conv2d(64, 128, 3)
          self.pool1 = nn.MaxPool2d((2,2))
          self.pool2 = nn.MaxPool2d((2,2))
          self.pool3 = nn.MaxPool2d((2,2))
          
          self.fc1 = nn.Linear(flatten_val, 512)
          self.fc2 = nn.Linear(512, num_classes)

     def forward(self, x):
          x = F.relu(self.conv1(x))
          x = self.pool1(x)
          x = F.relu(self.conv2(x))
          x = self.pool2(x)
          x = F.relu(self.conv3(x))
          x = self.pool3(x)
          x = x.flatten(start_dim=1)
          
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return F.softmax(x, dim=1)

num_classes = len([f.path for f in os.scandir( OS_path+"Images/") if f.is_dir()])


#Load model
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
if torch.cuda.is_available():
    checkpoint = torch.load(OS_path+load_mod)
else:
   checkpoint = torch.load(OS_path+load_mod, map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

net.eval()

print("\nModel ("+load_mod[6:]+") Loaded...")

classes = [ f.path[len("Images/"):] for f in os.scandir("Images/") if f.is_dir() ]
classes = sorted(classes, key=str.lower)
##CLASSIFICATION################################################
if CLASSIFY:
    #Metadata of Records
    df_recordings = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
    
    print("\nClassification: Started...")
    time.sleep(0.5)
    classify_setup = Classify()
    classify_setup.make_classify_data()

    classify_data = np.load("classify_data.npy", allow_pickle=True)

    ##Classification dataset
    test_classify = torch.Tensor(classify_data).view(-1, 13*HEIGHT_MULTIPLIER, 99)
    test_classify = test_classify/255.0

    files_all = next(walk(OS_path+"Images Classify/"))[2]
    files_wav = {}
    i = 0
    for f in files_all:
        if f[len(f)-3:] == 'png':
            files_wav[i] = f[:len(f)-4]
            i += 1
 
    file_names = []
    for c in range(len(files_wav)):
        file_names.append(files_wav[c])

    #First remove metadata of removed files
    index = df_recordings.index
    condition = ~df_recordings["filename"].isin(file_names)
    file_index = index[condition].tolist()
    df_recordings = df_recordings.drop(file_index)
    df_recordings.to_csv(OS_path+'recordings_metadata/recordings_metadata.csv', index=False)
    index = df_recordings.index

    #Reload
    df_recordings = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')

    data = {'File':file_names}
    df_classification = pd.DataFrame(data)  
    df_classification.set_index('File',inplace=True)
    time.sleep(0.5)
    with torch.no_grad():
        for i in tqdm(range(len(file_names))):
            net_out = net(test_classify[i].view(-1,1,13*HEIGHT_MULTIPLIER,99).to(device))[0]
           
            predicted_class = torch.argmax(net_out)
            df_classification.at[file_names[i], 'Prediction'] = classes[int(predicted_class)]
            
            #Find index and add prediction to metadata file
            condition = df_recordings["filename"] == file_names[i]
            file_index = index[condition].tolist()
            df_recordings.at[file_index, "classified"] = classes[int(predicted_class)]
    
    #Save recordings data
    df_recordings.to_csv(OS_path+'recordings_metadata/recordings_metadata.csv', index=False)
    time.sleep(0.5)
    print('\n')       
    print(df_classification)
    print("\nClassification: Finished...")

###################################################################