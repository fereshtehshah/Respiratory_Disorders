# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 02:01:54 2020

@author: sukris
"""

from tqdm import tqdm
import numpy as np

# This method gets data based on grouping and datatype.
def get_data(clips, grouping="default", dtype="clip"):
    
    # Return values based on keyword arguments
    if grouping == "default":
        return get_data_default(clips,dtype)
    #if grouping == "patient":
    #    return get_data_patient(clips,dtype)
    #if grouping == "chest location":
    #    return get_data_location(clips,dtype)
    if grouping == "recording equipment":
        return get_data_recording(clips,dtype)
    #if grouping == "aquisition mode":
    #    return get_data_aquisition(clips,dtype)

    # If input is invalid, return nothing
    print("Invalid input to get_data(type). Type can be a string with one of the following values:")
    print("patient")
    print("chest location")
    print("recording equipment")
    print("aquisition mode")
    return None

# This method splits the data into 4 classes: ["Normal","Wheeze","Crackle","Both"]
def get_data_default(clips,dtype):
    
    # Initialize output arrays
    data = [[],[],[],[]]
    
    # Loop through each clip and group the clips by wheeze/crackle detection
    for clip in tqdm(clips,"Grouping data by default"):
        # Get boolean values
        c = clip.crackle
        w = clip.wheeze
        
        # Append clip to corresponding list
        i = get_index(c,w)
        if dtype == "audio":
            data[i].append(clip.audio)
        elif dtype == "clip":
            data[i].append(clip)
        else:
            print("Found improper dtype:",dtype)
            print("Appropriate values for dtype are:",["audio","clip"])
            
    # Return the grouped data
    return data

# This method splits the data into 4 classes, grouped by recording equipment
def get_data_recording(clips,dtype):
    
    # Initialize output arrays
    # Each separate list represents clips recorded with the same recording equipment
    # Column 0: normal
    # Column 1: crackle
    # Column 2: wheeze
    # Column 3: both
    akgc417l = [[],[],[],[]]
    littc2se = [[],[],[],[]]
    litt3200 = [[],[],[],[]]
    meditron = [[],[],[],[]]
    
    # Loop through each clip and group them
    for clip in tqdm(clips,"Grouping data by recording equipment"):
        
        # Get information from clip
        c = clip.crackle
        w = clip.wheeze
        r = clip.rec_equipment
        
        # Assign index based on booolean wheeze/crackle information
        i = get_index(c,w)
        
        # Append clip to corresponding list based on dtype
        if dtype == "audio":
            if r == "AKGC417L":
                akgc417l[i].append(clip.audio)
            elif r == "LittC2SE":
                littc2se[i].append(clip.audio)
            elif r == "Litt3200":
                litt3200[i].append(clip.audio)
            elif r == "Meditron":
                meditron[i].append(clip.audio)
            else:
                print("Found improper recording equipment:",r)
        elif dtype == "clip":
            if r == "AKGC417L":
                akgc417l[i].append(clip)
            elif r == "LittC2SE":
                littc2se[i].append(clip)
            elif r == "Litt3200":
                litt3200[i].append(clip)
            elif r == "Meditron":
                meditron[i].append(clip)
            else:
                print("Found improper recording equipment:",r)
        else:
            print("Found improper dtype:",dtype, "in", clip.file_name)
            print("Appropriate values for dtype are:",["audio","clip"])
    
    return akgc417l, littc2se, litt3200, meditron

def split_data(classes,train=0,test=0,valid=0):
    # Check if the splits are correct
    if test + valid + train != 1:
        print("Splits don't add up to 100%")
        return None
    
    # Assign array lengths for test, valid, and train sets
    splits = np.zeros((3,len(classes)),dtype=int)
    i = 0
    for c in classes:
        total = len(c)
        splits[0,i] = int(train * total)
        splits[1,i] = int(test * total) + splits[0,i]
        splits[2,i] = total
        i += 1
    
    # Split the data set
    train_split = []
    test_split = []
    valid_split = []
    for n in range(len(classes)):
        train_split.append(classes[n][0:splits[0,n]])
        test_split.append(classes[n][splits[0,n]:splits[1,n]])
        valid_split.append(classes[n][splits[1,n]:splits[2,n]])

    return train_split, test_split, valid_split

def crop_clips(clips,seconds,sr):
    n_samples = int(seconds*sr)
    for clip in tqdm(clips,"Cropping clips"):
        audio = clip.sound_data
        if len(audio) < n_samples:
            clip.cropped_sound = np.pad(audio,(0,n_samples - len(audio)))
        elif len(audio) > n_samples:
            clip.cropped_sound = audio[:n_samples]
        else:
            clip.cropped_sound = audio
    
    return clips
    
def filter_clips(clips,lower,upper,sr):
    n_lower = lower * sr
    n_upper = upper * sr
    output = []
    for clip in tqdm(clips,"Filtering clips"):
        l = len(clip.sound_data)
        if l >= n_lower and l <= n_upper:
            output.append(clip)
    
    return crop_clips(output, lower)
        
def get_index(c,w):
    if c and w:
        return 3
    elif not c and not w:
        return 0
    elif c:
        return 1
    else:
        return 2