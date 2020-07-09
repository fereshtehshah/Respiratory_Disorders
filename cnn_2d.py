#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:12:27 2020

@author: elvanugurlu
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
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, Input, BatchNormalization,MaxPooling2D
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import History 
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

#%% Get clips (all at same sample rate for ease of use)
sr = 44100
#directory = "/Users/feres/SUMMER2020/ML_7641/Project/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
print("started")
#directory = "D:\\Google Drive\\Programs\\Jupyter\\Machine Learning\\project\\data\\audio_and_txt_files"
directory = "/Users/elvanugurlu/OneDrive - Georgia Institute of Technology/Classes/CS ML/Project/110374_267422_bundle_archive/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
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
train_split, test_split, valid_split = split_data(data,train=0.7,test=0.2,valid=0.1)

train_data = []
test_data = []
valid_data = []
train_labels = []
test_labels = []
valid_labels = []

i = 0
for clips in tqdm(train_split, "Training split"):
    for clip in clips:
        #train_data.append(clip.flattened_mfcc)
        train_data.append(clip.mfcc)
        train_labels.append(i)
    i += 1

i = 0      
for clips in tqdm(test_split,"Testing split"):
    for clip in clips:
        #test_data.append(clip.flattened_mfcc)
        test_data.append(clip.mfcc)
        test_labels.append(i)
    i += 1
i = 0
for clips in tqdm(valid_split,"Validation split"):
    for clip in clips:
        #valid_data.append(clip.flattened_mfcc)
        valid_data.append(clip.mfcc)
        valid_labels.append(i)
    i += 1
    
#%% Convert list to 3D array
train_data = np.asarray(train_data)
valid_data = np.asarray(valid_data)
test_data = np.asarray(test_data)

#%% Scaling the input to standardize features by removing the mean and scaling to unit variance
#scaler = preprocessing.StandardScaler().fit(train_data)
#train_data = scaler.transform(train_data)
#valid_data=scaler.transform(valid_data)
#test_data=scaler.transform(test_data)

#%% Applying PCA to reduce dimension while still keeping 99% of the original variance
#pca = decomposition.PCA(n_components=0.99, svd_solver = 'full') 
#pca.fit(train_data)
#train_data = pca.transform(train_data)   
#valid_data = pca.transform(valid_data) 
#test_data = pca.transform(test_data)
    
#%% Plots for visualization 
# # Plotting number of component vs explained variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

# # Scatter plot of the classes with most informative 3 principle components
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], c=train_labels[:,0])

#%% One-hot encoding of the labels
train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)

train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2],1)
valid_data = valid_data.reshape(valid_data.shape[0],valid_data.shape[1],valid_data.shape[2],1)
test_data =test_data.reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2],1)

def evaluate_2d_cnn_model(train_data ,train_labels, test_data,test_labels ,valid_data, valid_labels):
    ### Building the model
    hidden_num_units = 2048
    hidden_num_units1 = 1024
    hidden_num_units2 = 128
    output_num_units = train_labels.shape[1]
    
    epochs = 50 #10
    batch_size = 16 #16
    pool_size = (2, 2)
    input_shape = Input(shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    kernel_size = (3, 3)
    
    model = Sequential([
    
        Conv2D(16, kernel_size, activation='relu', input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]), padding='same'),
        BatchNormalization(),
    
        Conv2D(16, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
    
        Conv2D(32, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
    
        Conv2D(32, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
    
        Conv2D(64, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
    
        Conv2D(64, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
    
        Flatten(),
    
        Dense(units=hidden_num_units, activation='relu'),
        Dropout(0.3),
        Dense(units=hidden_num_units1, activation='relu'),
        Dropout(0.3),
        Dense(units=hidden_num_units2, activation='relu'),
        Dropout(0.3),
        Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'),
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    print("Model is compiled")
    model.summary()
    
    
    ### Training the model
    trained_model_conv = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                                   validation_data=(valid_data, valid_labels))
    print("Model is trained")
    
    ### Prdicting the class
    pred = model.predict_classes(test_data)
    
    ### Evaluating the model
    scores = model.evaluate(test_data, test_labels)
    
    print(model.metrics_names)
    print(scores)
    
    acc = trained_model_conv.history['accuracy']
    val_acc = trained_model_conv.history['val_accuracy']
    loss = trained_model_conv.history['loss']
    val_loss = trained_model_conv.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    return acc


 
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment( trainX, trainy, testX, testy,validX, validy, repeats=10):

	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_2d_cnn_model(trainX, trainy, testX, testy, validX, validy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
# run the experiment
run_experiment( train_data ,train_labels, test_data,test_labels ,valid_data, valid_labels, repeats=1)