# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 02:08:28 2020

@author: sukris
"""

from import_dataset import import_all_files
from group_data import get_data
import numpy as np

directory = "D:\Documents\GitHub\ML_Prj\data"

# Get clips
clips = import_all_files(directory)

# Get data and test separated only by class
data0 = get_data(clips, grouping="default", dtype="clip")

# Test if all clips in first group are normal (No wheezes or crackles)
for clip in data0[0]:
    if clip.crackle or clip.wheeze:
        print("Test failed, line 24",clip.crackle,clip.wheeze)
        break
for clip in data0[1]:
    if not (clip.crackle and not clip.wheeze):
        print("Test failed, line 28",clip.crackle,clip.wheeze)
        break
for clip in data0[2]:
    if not (not clip.crackle and clip.wheeze):
        print("Test failed, line 32",clip.crackle,clip.wheeze)
        break
for clip in data0[3]:
    if not (clip.crackle and clip.wheeze):
        print("Test failed, line 36",clip.crackle,clip.wheeze)
        break

# Get data and test separated by class and recording equipment
r0, r1, r2, r3 = get_data(clips, grouping="recording equipment", dtype="clip")

# Get grouping information
total_clips = len(clips)

total_normal = len(data0[0])
total_crackle = len(data0[1])
total_wheeze = len(data0[2])
total_both = len(data0[3])
total_data0 = total_normal + total_crackle + total_wheeze + total_both

count_r = np.array([[len(r0[0]), len(r0[1]), len(r0[2]), len(r0[3])],
                    [len(r1[0]), len(r1[1]), len(r1[2]), len(r1[3])],
                    [len(r2[0]), len(r2[1]), len(r2[2]), len(r2[3])],
                    [len(r3[0]), len(r3[1]), len(r3[2]), len(r3[3])]])
count_r_T = count_r.transpose()

# Test if groupings are accurate
print()
print()
print("Total number of clips:   ", len(clips))
print()
print("Number of normal clips:  ", total_normal)
print("Number of crackle clips: ", total_crackle)
print("Number of wheeze clips:  ", total_wheeze)
print("Number of both clips:    ", total_both)
print("Total check:             ", total_clips == total_data0)
print()
print("Number of AKGC417L: ", np.sum(count_r[0]))
print("Number of LittC2SE: ", np.sum(count_r[1]))
print("Number of Litt3200: ", np.sum(count_r[2]))
print("Number of Meditron: ", np.sum(count_r[3]))
print("Class check - norm: ", np.sum(count_r_T[0]) == total_normal)
print("Class check - crac: ", np.sum(count_r_T[1]) == total_crackle)
print("Class check - whee: ", np.sum(count_r_T[2]) == total_wheeze)
print("Class check - both: ", np.sum(count_r_T[3]) == total_both)
print("Total check:        ", total_clips = np.sum(count_r))