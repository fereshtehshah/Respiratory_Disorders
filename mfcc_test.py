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
        clip.mfcc = mfcc(y=clip.sound_data, sr=clip.sr)

#%% Plot random mfccs from each group
c = 0
for group in data:
    # Get images to plot
    key = np.random.randint(0,len(group),size=4)
    arr = []
    for n in range(4):
        arr.append(group[key[n]])
    
    plt.figure(dpi=500)

    for n in range(4):
        clip = group[key[n]]
        plt.subplot(2,2,n+1)
        librosa.display.specshow(clip.mfcc, x_axis='time', y_axis='mel', sr=clip.sr)
        plt.title(clip.recording)
    
    plt.tight_layout(pad=3.0)
    
    plt.show()