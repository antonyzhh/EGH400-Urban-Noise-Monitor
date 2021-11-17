import os
from os import walk
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib
from pathlib import Path
matplotlib.use('Agg')

import directory

OS_path = directory.path()

#This function loads and combines the two dataframes into one
def dataframe():
    df_recordings = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')
    df_training = pd.read_csv(OS_path+'recordings_metadata/recordings_training_log.csv')

    frames = [df_recordings, df_training]
    df = pd.concat(frames)
    df.index = range(len(df))

    return df

#Total classification per class
def class_total_classifications():
    #Load the classified audio log file
    df = dataframe()
    classified_classes = df.classified.unique()
    count = np.zeros(len(classified_classes))

    for i in range(len(classified_classes)):
        index = df.index
        condition = df["classified"] == classified_classes[i]
        file_index = index[condition].tolist()
        count[i] = len(file_index)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.bar(classified_classes,count)
    plt.xticks(fontsize=5)
    ax.set_ylabel('Count')
    ax.set_xlabel('Classes')
    ax.set_title('Total classifications per Class')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    intervals = float(1)
    loc = plticker.MultipleLocator(base=intervals)
    ax.yaxis.set_major_locator(loc)
    plt.savefig('static/graph1.png')

def year_total_classifications():
    df = dataframe()
    year = df.year.unique()
    year = range(min(year),max(year)+1)
    count = np.zeros(len(year))

    for i in range(len(year)):
        index = df.index
        condition = df["year"] == year[i]
        file_index = index[condition].tolist()
        count[i] = len(file_index)

    #fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.plot(year,count)
    ax.plot(year,count,'r*')
    #plt.xticks(fontsize=8)
    ax.set_ylabel('Count')
    ax.set_xlabel('Year')
    ax.set_title('Number of classifications per year')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.set_xticks(year)
    ax.set_ylim(ymin=0)
    #plt.show()
    plt.savefig('static/graph2.png')

#Number of specificied class classifications per year
def class_year_classifications(classified_class):
#classified_class = 'children_playing'
    df = dataframe()
    index = df.index
    condition = df["classified"] == classified_class
    file_index = index[condition].tolist()

    year = []
    for i in file_index:
        year.append(df["year"][i])

    year_range =  range(min(year), max(year)+1)#list(set(year))
    count = np.zeros(len(year_range))

    for i in range(len(year)):
        index = year_range.index(year[i])
        count[index] += 1

    #fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.bar(year_range,count)
    #plt.xticks(fontsize=8)
    ax.set_ylabel('Count')
    ax.set_xlabel('Year')
    ax.set_title('Number of "'+classified_class+'" classifications per year')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.set_xticks(year_range)
    #plt.show()
    plt.savefig('static/graph3.png')