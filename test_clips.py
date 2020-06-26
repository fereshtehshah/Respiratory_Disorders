# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:57:18 2020

@author: sukris
"""

#%% Import libraries
from import_dataset import import_all_files
from group_data import get_data
import numpy as np

directory = "D:\\Google Drive\\Programs\\Jupyter\\Machine Learning\\project\\data\\audio_and_txt_files"

#%% Get clips
clips = import_all_files(directory)

#%% Check sample rates
sample_rates = {}
for clip in clips:
    key = str(clip.sr)
    if key in sample_rates:
        sample_rates[key] += 1
    else:
        sample_rates[key] = 0
        
#%% Plot clip lengths
times = []
for clip in clips:
    sr = clip.sr
    n = len(clip.sound_data)
    t = n/sr
    
    times.append(t)

times = np.array(times)
plt.figure()
plt.plot(times,'.')
plt.xlabel("Clip Number")
plt.ylabel("Time in Seconds")
plt.title("Plot of Clip Lengths")
    