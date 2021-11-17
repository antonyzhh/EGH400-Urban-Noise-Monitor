import os
from os import walk
import pandas as pd

import directory

OS_path = directory.path()

def model_show():
    try:
        os.remove(OS_path+"MODEL/.DS_Store")
    except:
        print(".DS_Store file not found")

    files_all = next(walk(OS_path+"MODEL/"))[2]
    files_all.sort()

    return files_all

def model_select(filename):
    df = pd.DataFrame({'active model': [filename]})
    df.to_csv(OS_path+'active_model.csv', index=False)

def active_model():
    df = pd.read_csv(OS_path+'active_model.csv')
    model_name = df["active model"][0]

    return model_name


