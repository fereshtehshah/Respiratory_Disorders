# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:42:18 2020

@author: sukris
"""
#%% Import libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from import_dataset import import_all_files
from group_data import get_data
from librosa.feature import mfcc
from librosa.feature import melspectrogram
from tqdm import tqdm

directory = "D:\\Google Drive\\Programs\\Jupyter\\Machine Learning\\project\\data\\audio_and_txt_files"

#%% Get clips
clips = import_all_files(directory)

#%% Get data and test separated only by class
data = get_data(clips, grouping="default", dtype="clip")

#%% Do mfcc on every clip
c = 0
images = [[],[],[],[]]
for group in data:
    for clip in tqdm(group, "Taking MFCC of clips in group " + str(c) + " of " + str(len(data))):
        images[c].append(mfcc(y=clip.sound_data, sr=clip.sr))
    c += 1

#%% Plot random mfccs from each group
c = 0
for group in images:
    # Get images to plot
    key = np.random.randint(0,len(group),size=4)
    
    plt.figure(dpi=500)

    plt.subplot(2,2,1)
    librosa.display.specshow(group[key[0]], x_axis='time', y_axis='mel')
    plt.title(data[c][key[0]].recording)
    
    plt.subplot(2,2,2)
    librosa.display.specshow(group[key[1]], x_axis='time', y_axis='mel')
    plt.title(data[c][key[1]].recording)
    
    plt.subplot(2,2,3)
    librosa.display.specshow(group[key[2]], x_axis='time', y_axis='mel')
    plt.title(data[c][key[2]].recording)
    
    plt.subplot(2,2,4)
    librosa.display.specshow(group[key[3]], x_axis='time', y_axis='mel')
    plt.title(data[c][key[3]].recording)
    
    plt.tight_layout(pad=3.0)
    
    plt.show()