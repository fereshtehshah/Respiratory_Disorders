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
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing

#%% Get clips (all at same sample rate for ease of use)
sr = 44100
directory = "/Users/feres/SUMMER2020/ML_7641/Project/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
print("started")
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
train_split, test_split, valid_split = split_data(data,train=0.8,test=0.2,valid=0.0)

train_data = []
test_data = []
valid_data = []
train_labels = []
test_labels = []
valid_labels = []

i = 0
for clips in tqdm(train_split, "Training split"):
    for clip in clips:
        train_data.append(clip.flattened_mfcc)
        train_labels.append(i)
    i += 1

i = 0      
for clips in tqdm(test_split,"Testing split"):
    for clip in clips:
        test_data.append(clip.flattened_mfcc)
        test_labels.append(i)
    i += 1
i = 0
for clips in tqdm(valid_split,"Validation split"):
    for clip in clips:
        valid_data.append(clip.flattened_mfcc)
        valid_labels.append(i)
    i += 1
params_grid = [{'kernel': ['linear'], 'C': [1]}]
encoder = preprocessing.LabelEncoder()
encoder.fit(train_labels)
Y_train = encoder.transform(train_labels)
svm_model = GridSearchCV(SVC(), params_grid, cv=10)
svm_model.fit(train_data, Y_train)

final_model = svm_model.best_estimator_
Y_pred = final_model.predict(test_data)
print(Y_pred)
Y_pred_label = list(encoder.inverse_transform(Y_pred))
print(Y_pred_label)
print(confusion_matrix(test_labels,Y_pred_label))
print("\n")
print(classification_report(test_labels,Y_pred_label))
#%% Applying PCA
# pca = decomposition.PCA(n_components=0.2, svd_solver = 'full') 
# pca.fit(train_data)
# data_matrix_new = pca.transform(train_data)    

# params_grid = [{'kernel': ['linear'], 'C': [1]}]

# encoder = preprocessing.LabelEncoder()
# encoder.fit(train_labels)
# Y_train = encoder.transform(train_labels)
# svm_model = GridSearchCV(SVC(), params_grid, cv=5)
# svm_model.fit(data_matrix_new, Y_train)

# final_model = svm_model.best_estimator_
# Y_pred = final_model.predict(test_data)
# Y_pred_label = list(encoder.inverse_transform(Y_pred))
# print(confusion_matrix(test_labels,Y_pred_label))
# print("\n")
# print(classification_report(test_labels,Y_pred_label))


# # Plotting number of component vs explained variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

'''
# Scatter plot of the classes with most informative 3 principle components
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_matrix_new[:, 0], data_matrix_new[:, 1], data_matrix_new[:, 2], c=labels[:,0])
'''