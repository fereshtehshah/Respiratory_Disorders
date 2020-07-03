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
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for 3D plotting

#%% Get clips (all at same sample rate for ease of use)
sr = 44100
#directory = "D:\\Google Drive\\Programs\\Jupyter\\Machine Learning\\project\\data\\audio_and_txt_files"
directory = "/Users/elvanugurlu/OneDrive - Georgia Institute of Technology/Classes/CS ML/Project/110374_267422_bundle_archive/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
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
    clip.flattened_mfcc = clip.mfcc.flatten()

#%% Separate data by class
data = get_data(clips, grouping="default", dtype="clip")

#%% Generate data matrix and label array
# Label convention:  
# 0: no crackle, no wheeze 
# 1: crackle, no wheeze
# 2: no crackle, wheeze 
# 3: crackle and wheeze
data_matrix = np.zeros((len(clips), clips[0].flattened_mfcc.shape[0]))  # dimension: number of recordings x total mfcc coefficients
labels = np.zeros((len(clips),1))

index = 0
for clip in tqdm(clips,"Generating data matrix"):
    data_matrix[index,:]  = clip.flattened_mfcc
    if not clip.crackle and not clip.wheeze:
        labels[index] = 0
    elif clip.crackle and not clip.wheeze:
        labels[index] = 1
    elif not clip.crackle and  clip.wheeze:
        labels[index] = 2 
    else:
        labels[index] = 3
    index = index + 1

#%% Applying PCA
pca = decomposition.PCA(n_components=0.99, svd_solver = 'full') 
pca.fit(data_matrix)
data_matrix_new = pca.transform(data_matrix)    

# Plotting number of component vs explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# Scatter plot of the classes with most informative 3 principle components
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_matrix_new[:, 0], data_matrix_new[:, 1], data_matrix_new[:, 2], c=labels[:,0])

    
