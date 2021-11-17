import os
from os import walk
import pandas as pd
import time
#import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play
import shutil

import directory

OS_path = directory.path()

def select():
    files_all = next(walk(OS_path+"Audio Classify/"))[2]

    return files_all

def playsound(val):
    files_all = select()
    filename = OS_path+'Audio Classify/'+files_all[int(val)]

    # wave_obj = sa.WaveObject.from_wave_file(filename)
    # play_obj = wave_obj.play()
    # play_obj.wait_done()  # Wait until sound has finished playing
    clip = AudioSegment.from_wav(filename)
    play(clip)

    files_all = select()
    df_recordings = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
    index = df_recordings.index
    condition = df_recordings["filename"] == files_all[int(val)]
    file_index = index[condition].tolist()[0]

    if not pd.isna(df_recordings.actual[file_index]):
        info = ["checked",df_recordings.classified[file_index], df_recordings.actual[file_index]]
        classes = []

        return info, classes
    else:
        classes = [ f.path[len("Images/"):] for f in os.scandir("Images/") if f.is_dir() ]
        classes = sorted(classes, key = str.lower)
        info = ["unchecked",df_recordings.classified[file_index], df_recordings.actual[file_index]]

        return info, classes

def actual_conf(val, filename):
    classes = [ f.path[len("Images/"):] for f in os.scandir("Images/") if f.is_dir() ]
    classes = sorted(classes, key = str.lower)

    df_recordings = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
    index = df_recordings.index
    condition = df_recordings["filename"] == filename
    file_index = index[condition].tolist()[0]

    if val == "None of the above":
        df_recordings.at[file_index, "actual"] = "unknown"
        info = [df_recordings["filename"][file_index], 'unknown']
    else:
        df_recordings.at[file_index, "actual"] = classes[int(val)]
        info = [df_recordings["filename"][file_index], classes[int(val)]]

    df_recordings.to_csv(OS_path+'recordings_metadata/recordings_metadata.csv', index=False)
    
    return info

def move(which_class, filename):
    df_recordings = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
    index = df_recordings.index
    condition = df_recordings["filename"] == filename
    file_index = index[condition].tolist()[0]

    shutil.move(OS_path+'Images Classify/'+filename+'.png', OS_path+'Images/'+which_class+'/'+filename+'.png')
    shutil.move(OS_path+'Audio Classify/'+filename, OS_path+'Recorded Audio Files Training/'+filename)
    print('--> File moved to the "'+which_class+'" training folder')

    #Update Log
    df_recordings_training = pd.read_csv(OS_path+'recordings_metadata/recordings_training_log.csv')
    df_recordings_training = df_recordings_training.append(df_recordings.iloc[file_index])
    df_recordings = df_recordings.drop(file_index)
    df_recordings_training.to_csv(OS_path+'recordings_metadata/recordings_training_log.csv', index=False)
        
def delete_show():
    files_all = next(walk(OS_path+"Audio Classify/"))[2]

    return files_all

def delete(filename):
    try:
        df_recordings = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
        index = df_recordings.index
        condition = df_recordings["filename"] == filename
        file_index = index[condition].tolist()[0]
        df_recordings = df_recordings.drop(file_index)
        df_recordings.to_csv(OS_path+'recordings_metadata/recordings_metadata.csv', index=False)
    except:
        print("No metadata for this file")

    os.remove(OS_path+'Audio Classify/'+filename)

    try:
        os.remove(OS_path+'Images Classify/'+filename+'.png')
    except:
        print("File "+filename+".png does not exist")        