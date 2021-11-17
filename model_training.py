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
import confusion_matrix
import directory

OS_path = directory.path()

##CONTROLS
GENERATE_IMGS = False #For Classification
REBUILD_DATA = False

# MAKE and SAVE MODEL
MAKE_N_SAVE_MODEL = True


#Training Parameters
BATCH_SIZE = 32
EPOCHS = 65

##Generate Images to Classifiy
if GENERATE_IMGS and MAKE_N_SAVE_MODEL:
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

class SoundType():
    RESIZE_WIDTH = 99
    RESIZE_HEIGHT = 13*HEIGHT_MULTIPLIER

    class_path = [ f.path for f in os.scandir("Images/") if f.is_dir() ]
    class_path = sorted(class_path, key = str.lower)
    LABELS = {}

    i = 0
    for c in class_path:
        LABELS[c] = i
        i += 1

    training_data = []

    def make_training_data(self):
        class_type = []
        class_count = np.zeros(len(self.LABELS),dtype=object)
        i = 0
        for label in self.LABELS:
            class_type.append(label[len("Images/"):])
            print(label[len("Images/"):]+': ')
            time.sleep(0.3)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
                    self.training_data.append([np.array(img), np.eye(len(self.LABELS))[self.LABELS[label]]])
                    
                    for c in range(len(self.class_path)):
                        if label == self.class_path[c]:
                            class_count[c] += 1

                except Exception as e:
                    pass
            
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
 

        #CHECK DATA BALANCE  
        data = {}
        i = 0
        for d in self.class_path:
            data[d[len('Images/'):]] = class_count[i]
            i += 1

        # Create DataFrame  
        class_dist = pd.Series(data)  

        fig, ax = plt.subplots()
        ax.set_title('Class Distribution', y=1.08)
        ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
        ax.axis('equal')
        plt.show()

        total_classes = str(len(self.LABELS))
        total_data = str(len(self.training_data))
        print("DATA_SIZE: "+total_data+" files --> "+total_classes+" classes")


#RUN CODES
if REBUILD_DATA:
    print("Building training data...")
    time.sleep(0.5)

    soundtype = SoundType()
    soundtype.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)

#Data load examples
#plt.imshow(training_data[0][0], cmap="gray")
#training_data[0][1]
#plt.show()

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

num_classes = len([f.path for f in os.scandir( "Images/") if f.is_dir()])

if MAKE_N_SAVE_MODEL:
    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    loss_function = nn.MSELoss()

    X = torch.Tensor([i[0] for i in training_data]).view(-1, 13*HEIGHT_MULTIPLIER, 99)
    X = X/255.0
    y = torch.Tensor([i[1] for i in training_data])

    VAL_PCT = 0.1
    val_size = int(len(X)*VAL_PCT)
    #print(val_size)

    #Training Dataset
    train_X = X[:-val_size] #Image Data
    train_y = y[:-val_size] #Class label

    #Testing Dataset
    test_X = X[-val_size:]
    test_y = y[-val_size:]

    #print(len(train_X)+len(test_X))

    train_size = str(len(train_X))
    print("\nTRAINING_DATA_SIZE: "+train_size+" files")
    print("Model Training: Started...")
    time.sleep(0.5)

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            #print(i, i+BATCH_SIZE)
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,13*HEIGHT_MULTIPLIER,99).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
 
    print("Model Training: Finished...")
    time.sleep(0.5)

    print("\nModel Testing: Started...")
    time.sleep(0.5)
    #print(loss)

    correct_class = np.zeros(num_classes,dtype=object)
    total_class = np.zeros(num_classes,dtype=object)
    test_cm = np.zeros([num_classes,num_classes])
    arr_real = np.zeros(len(test_X))
    arr_predic = np.zeros(len(test_X))

    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1,1,13*HEIGHT_MULTIPLIER,99).to(device))[0]
            predicted_class = torch.argmax(net_out)

            #Confusion Matrix Data
            arr_real[i] = real_class.cpu().numpy()
            arr_predic[i] = predicted_class.cpu().numpy()
            test_cm[predicted_class, real_class] += 1

            if predicted_class == real_class:
                correct_class[int(real_class)] += 1
                correct += 1
            total += 1
            total_class[int(real_class)] += 1

    accuracy_class = [x/y*100 for x, y in zip(correct_class, total_class)]

    classes = [ f.path[len("Images/"):] for f in os.scandir("Images/") if f.is_dir() ]
    classes = sorted(classes, key = str.lower)
    i = 0
    for c in classes:
        print(int(accuracy_class[i]),'% --> '+c)
        i += 1

    avg_acc = str(round(correct/total*100,3))
    print("\nAverage Accuracy:"+avg_acc+"%")

    #confusion_matrix.main(arr_real.astype(int), arr_predic.astype(int), classes)
    confusion_matrix.simple(test_cm, classes, total_class)

    print("Model Testing: Finished...")

    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S-acc-")
    #torch.save(net.state_dict(), OS_path+"MODEL/mod-"+date_time+avg_acc+".pth")
    torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, OS_path+"MODEL/mod-"+date_time+avg_acc+".pth")
    print("Model Saved as... mod-"+date_time+avg_acc+".pth")

else:
    print("Nothing done...")