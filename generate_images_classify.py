##ENV-> source activate /Users/Antony_/opt/anaconda3/envs/snowflakes
## activate C:\Users\anton\anaconda3\envs\snowflakes
import os
from os import walk
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
from pathlib import Path
import librosa

import directory

OS_path = directory.path()

##FUNCTIONS############################################################
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

##CODE##
REBUILD_CLASSIFY = True
REBUILD_TRAINING = False

if REBUILD_TRAINING:
    #URBANSOUND8K###########################################################
    df = pd.read_csv(OS_path+'UrbanSound8K/metadata/UrbanSound8K.csv')
    df.set_index('slice_file_name', inplace=True)
    df.rename({'class':'label'}, axis=1, inplace=True)

    for f in tqdm(df.index):
        my_file = Path(OS_path+'UrbanSound8K/audio/fold_all_vs/'+f)
        if my_file.is_file():
            rate, signal = wavfile.read(OS_path+'UrbanSound8K/audio/fold_all_vs/'+f)
            df.at[f, 'length'] = signal.shape[0]/rate

    #Remove short audio files, street music class and gun shot class
    index = 0
    while index < len(df):
        if df.length[index] < 0.5 or df.label[index] == 'street_music' or df.label[index] == 'gun_shot':
            my_file = Path(OS_path+'UrbanSound8K/audio/fold_all_vs/'+df.index[index])
            if my_file.is_file():
                os.remove(OS_path+'UrbanSound8K/audio/fold_all_vs/'+df.index[index])
            df = df.drop(df.index[index]) 
        elif np.isnan(df.length[index]):
            df = df.drop(df.index[index])
        else:
            index += 1   
    
    classes = list(np.unique(df.label))

    signals_8K = {}
    mfccs_8K = {}
    classID_8K = {}

    for c in tqdm(df.index):
        #wav_file = df[df.index == c].iloc[0,0]
        signal, rate = librosa.load(OS_path+'UrbanSound8K/audio/fold_all_vs/'+c, sr=44100)
        mask = envelope(signal, rate, 0.0005) 
        signals_8K[c] = signal[mask]
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs_8K[c] = mel
        classID_8K[c] = df.classID[c]

    ##Music Dataset#######################################
    signals_music = {}
    mfccs_music = {}

    path_music = OS_path+'Music/music_all/'
    no_music = len(os.listdir('Music/music_all/'))
    all_music = next(walk(path_music))[2]

    for c in tqdm(range(no_music)):
        #wav_file = df[df.index == c].iloc[0,0]
        signal, rate = librosa.load(path_music+all_music[c], sr=44100)
        mask = envelope(signal, rate, 0.0005) 
        signals_music[all_music[c]] = signal[mask]
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs_music[all_music[c]] = mel

    ##AOB data#############################################################
    signals_AOB = {}
    mfccs_AOB = {}

    path_AOB = OS_path+'AOB/'
    folders_AOB = [ f.path[len('AOB/'):] for f in os.scandir("AOB/") if f.is_dir() ]
    no_AOB = np.zeros(len(folders_AOB))
    for i in range(len(folders_AOB)):
        no_AOB = len(os.listdir('AOB/'+folders_AOB[i]))
        all_AOB = next(walk(path_AOB+folders_AOB[i]))[2]

        for c in tqdm(range(no_AOB)):
            signal, rate = librosa.load(path_AOB+'/'+folders_AOB[i]+'/'+all_AOB[c], sr=44100)
            mask = envelope(signal, rate, 0.0005) 
            signals_AOB[all_AOB[c]] = signal[mask]
            mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
            mfccs_AOB[all_AOB[c]] = mel

#Classification Wav Files
signals = {}
mfccs = {}
classID = {}

files_all = next(walk(OS_path+"Audio Classify/"))[2]
files_wav = {}
i = 0
for f in files_all:
    if f[len(f)-3:] == 'wav':
        files_wav[i] = f
        i += 1

for f in tqdm(range(len(files_wav))):
    signal, rate = librosa.load(OS_path+"Audio Classify/"+files_wav[f], sr=44100)
    mask = envelope(signal, rate, 0.0005) 
    signals[files_wav[f]] = signal[mask]
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[files_wav[f]] = mel

# Save Images
cmap = plt.cm.hot

if REBUILD_TRAINING:
    # Save_Training
    if len(os.listdir('Images')) == 0:
        for c in classes:
            os.makedirs('Images/'+c)

        #Save UrbanSound8K imgs
        i = 0    
        for f in tqdm(df.index):
            # save the image
            data = list(mfccs_8K.values())[i]
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            plt.imsave('Images/'+classes[list(classID_8K.values())[i]]+'/'+f[:len(f)-3]+'png', cmap(norm(data)))
            i += 1
        print('\nUrbanSound8K Images generated')

        #Save GITZAN dataset music imgs
        if os.path.isdir('Images/music/') == False:
            os.makedirs('Images/music/')
        i = 0
        for c in tqdm(range(no_music)):
            data = list(mfccs_music.values())[i]
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            plt.imsave('Images/music/'+all_music[c][:len(all_music[c])-3]+'png', cmap(norm(data)))
            i += 1
        print('\nMusic Images generated')

        #Save AOB imgs
        i = 0
        for l in range(len(folders_AOB)):
            no_AOB = len(os.listdir('AOB/'+folders_AOB[l]))
            all_AOB = next(walk(path_AOB+folders_AOB[l]))[2]
            for c in tqdm(range(no_AOB)):
                data = list(mfccs_AOB.values())[i]
                norm = plt.Normalize(vmin=data.min(), vmax=data.max())
                plt.imsave('Images/'+folders_AOB[l]+'/'+all_AOB[c][:len(all_AOB[c])-3]+'png', cmap(norm(data)))
                i += 1
        print('\nAOB Images generated')


# Save_Classify
if os.path.isdir('Images Classify/') == False:
    os.makedirs('Images Classify/')

if len(os.listdir('Images Classify')) >= 1:
    if REBUILD_CLASSIFY:
        shutil.rmtree('Images Classify/')
        os.makedirs('Images Classify/')
        print("\nEmptied Classification Images Folder...")

if len(os.listdir('Images Classify')) == 0:
    i = 0
    for f in tqdm(range(len(files_wav))):
        # save the image
        data = list(mfccs.values())[i]
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        plt.imsave('Images Classify/'+files_wav[f]+'.png', cmap(norm(data)))
        i += 1
    print('\nNew Classification Images generated...')
else:
    print('\nNew Classification Images not generated...')