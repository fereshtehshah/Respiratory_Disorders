# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:00:15 2020

@author: sukris
"""

#%% Import libraries
import numpy as np
from import_dataset import import_all_files
from group_data import get_data
from librosa.feature import mfcc
from tqdm import tqdm

#%% Get clips (all at same sample rate for ease of use)
sr = 44100
directory = "D:\\Google Drive\\Programs\\Jupyter\\Machine Learning\\project\\data\\audio_and_txt_files"
clips = import_all_files(directory,sr)

#%% Do mfcc on zero-padded audio

# Get max audio length
max_len = len(clips[0].sound_data)
for clip in clips:
    l = len(clip.sound_data)
    if max_len < l:
        max_len = l

# Do MFCC
for clip in tqdm(clips,"Doing MFCC"):
    n = max_len - len(clip.sound_data)
    clip.mfcc = mfcc(y=np.pad(clip.sound_data,(0,n)), sr=sr)

#%% Separate data by class
data = get_data(clips, grouping="default", dtype="clip")