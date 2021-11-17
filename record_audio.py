#from os import path
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import pandas as pd
import numpy as np

import directory

OS_path = directory.path()

def record(duration):
    fs = 48000  # Sample rate
    seconds = duration  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    print('Recording...')
    sd.wait()  # Wait until recording is finished

    now = datetime.now() # current date and time
    date_time = now.strftime("%d-%m-%Y_%H;%M;%S")
    filepath = OS_path+'Audio Classify/recording_'+date_time+'.wav'
    myrecording = np.vstack((myrecording.transpose()[0], myrecording.transpose()[0]))
    myrecording_stereo = myrecording.transpose()
    write(filepath, fs, myrecording_stereo)  # Save as WAV file

    print('Saved as -> recording_'+date_time+'.wav')

    #Add metadata of new audio recording
    df = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
    df2 = {'filename': 'recording_'+date_time+'.wav',
        'day': date_time[:2], 
        'month': date_time[3:5],
        'year': date_time[6:10],
        'time': date_time[11:13]+':'+date_time[14:16],
        'classified': 'to be classified'}
    df = df.append(df2, ignore_index = True)
    #Save
    df.to_csv(OS_path+'recordings_metadata/recordings_metadata.csv', index=False)

    filename = 'recording_'+date_time+'.wav'
    filepath_out = OS_path+'Audio Classify/'
    return filename, filepath_out

def upload_save(filename, date, time):
    #Add metadata of new audio recording
    df = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
    df2 = {'filename': filename,
        'day': int(date[8:]), 
        'month': int(date[5:7]),
        'year': int(date[:4]),
        'time': time,
        'classified': 'to be classified'}
    df = df.append(df2, ignore_index = True)
    #Save
    df.to_csv(OS_path+'recordings_metadata/recordings_metadata.csv', index=False)
