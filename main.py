# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:00:15 2020

@author: sukris
"""

#%% Import libraries
import numpy as np
from import_dataset import import_all_files
from group_data import get_data
from group_data import split_data
from group_data import crop_clips
from group_data import filter_clips
from librosa.feature import mfcc
from tqdm import tqdm
from sklearn import decomposition
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D #for 3D plotting

#%% Get clips (all at same sample rate for ease of use)
sr = 44100
directory = "C:\\Users\\Sukris\\Google Drive\\Programs\\Jupyter\\Machine Learning\\project\\data\\audio_and_txt_files"
#directory = "D:\\Google Drive\\Programs\\Jupyter\\Machine Learning\\project\\data\\audio_and_txt_files"
#directory = "/Users/elvanugurlu/OneDrive - Georgia Institute of Technology/Classes/CS ML/Project/110374_267422_bundle_archive/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
clips = import_all_files(directory,sr)

# %% Crop/filter clips
clips = crop_clips(clips,5,sr)
# clips = filter_clips(clips,5,6,sr)

#%% Do mfcc on cropped audio
for clip in tqdm(clips,"Doing MFCC"):
    clip.mfcc = mfcc(y=clip.cropped_sound, sr=sr)
    clip.flattened_mfcc = clip.mfcc.flatten()

#%% Separate data by class
data = get_data(clips, grouping="default", dtype="clip")

#%% Split data into training, testing, and validation sets/labels
for d in data:
    random.shuffle(d)
train_split, test_split, valid_split = split_data(data,train=0.6,test=0.2,valid=0.2)

train_data = []
test_data = []
valid_data = []
train_labels = []
test_labels = []
valid_labels = []

for clips in train_split:
    i = 0
    for clip in clips:
        train_data.append(clip.flattened_mfcc)
        train_labels.append(i)
        i += 1
        
for clips in test_split:
    i = 0
    for clip in clips:
        test_data.append(clip.flattened_mfcc)
        test_labels.append(i)
        i += 1

for clips in valid_split:
    i = 0
    for clip in clips:
        valid_data.append(clip.flattened_mfcc)
        valid_labels.append(i)
        i += 1

#%% Applying PCA
pca = decomposition.PCA(n_components=0.2, svd_solver = 'full') 
pca.fit(train_data)
data_matrix_new = pca.transform(train_data)    

# Plotting number of component vs explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

'''
# Scatter plot of the classes with most informative 3 principle components
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_matrix_new[:, 0], data_matrix_new[:, 1], data_matrix_new[:, 2], c=labels[:,0])
'''