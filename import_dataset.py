# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:34:13 2020
@author: Serhat
"""
import numpy as np
import librosa
from os import listdir
from os.path import isfile, join, splitext
import os.path
from tqdm import tqdm


class Recording:
    def __init__(self, filename=None, sr=None, data=None):
        self.filename = splitext(filename)[0]+'.wav' if filename else None
        self._sr = sr
        self._data = data

    @property
    def data(self):
        if self._data is None:
            if self.filename is None:
                return None

            self._data, self._sr = librosa.load(self.filename, sr=self.sr,
                                                mono=False, dtype=np.float32)
        return self._data, self._sr

    @property
    def sr(self):
        return self._sr

    def __str__(self):
        return splitext(os.path.split(self.filename)[1])[0]


class Clip:
    def __init__(self, recording, patient_id, rec_i, chest_loc, acq_mode,
                 rec_equipment, crackle, wheeze, start_t=None, end_t=None):

        self.recording = recording
        self.patient_id = patient_id if isinstance(patient_id, int) else int(patient_id) 
        self.rec_i = rec_i
        self.chest_loc = chest_loc
        self.acq_mode = acq_mode
        self.rec_equipment = rec_equipment
        self.crackle = crackle
        self.wheeze = wheeze
        self.start_t = start_t
        self.end_t = end_t

    def __str__(self):
        return "Clip({}_{}_{}_{}_{}, c={}, w={}, ({:0.3f}s, {:0.3f}s))".format(self.patient_id, self.rec_i, self.chest_loc, self.acq_mode, self.rec_equipment, int(self.crackle), int(self.wheeze), self.start_t, self.end_t)

    def __repr__(self):
        return self.__str__()

    @property
    def sr(self):
        sr = self.recording.sr if self.recording else None
        return sr
    
    def recording_name(self):
        return str(self.recording)
        
    @property
    def sound_data(self):
        (sound_data, sr) = self.recording.data
        if self.start_t is None:
            self.start_t = 0
        if self.end_t is None:
            self.end_t = librosa.samples_to_time(sound_data.size, sr)

        start_i = librosa.time_to_samples(self.start_t, sr)
        end_i = librosa.time_to_samples(self.end_t, sr)

        return sound_data[start_i:end_i]


    @staticmethod
    def parse_annotations(filename):
        # Remove extension in case there is one
        filename = splitext(filename)[0]
        annotations = []
        with open("{}.txt".format(filename)) as f:
            for line in f:
                annotations.append([float(num) if i < 2 else bool(int(num))
                                    for i, num in enumerate(line.split())])

        return annotations

    @classmethod
    def generate_from_file(cls, filename, sr=None, lazy=False):
        # Remove extension in case there is one
        filename_ = splitext(filename)[0]
        if not lazy:
            (sound_data, sr) = librosa.load("{}.wav".format(filename_),
                                            sr=sr, mono=False,
                                            dtype=np.float32)
            recording = Recording(filename, sr, sound_data)
        else:
            recording = Recording(filename, sr)

        annotations = cls.parse_annotations("{}".format(filename_))
        metadata = tuple(os.path.split(filename_)[-1].split("_"))

        clips = []
        for a in annotations:
            clips.append(cls(recording, patient_id=metadata[0],
                             rec_i=metadata[1], chest_loc=metadata[2],
                             acq_mode=metadata[3], rec_equipment=metadata[4],
                             crackle=a[2], wheeze=a[3],
                             start_t=a[0], end_t=a[1]))
        return clips


def import_all_files(directory, sr=None, lazy=False):
    # Get a set of files (to prevent duplicates) in the directory
    # without the extension
    filenames = set(splitext(f)[0].split()[0] for f in listdir(directory)
                    if isfile(join(directory, f)))

    # This produces a list of lists of clips
    clips = [Clip.generate_from_file(join(directory, f), sr=sr, lazy=lazy)
             for f in tqdm(filenames, "Files to Clips")]

    # This flattens it into a single list of clips
    clips = [item for sublist in tqdm(clips) for item in sublist]

    return clips


if __name__ == "__main__":
    # execute only if run as a script
    directory = "C:\\Users\\Serhat\\OneDrive - Georgia Institute of Technology\\Classes\\CS 7641\\project\\110374_267422_bundle_archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files"
    clips = import_all_files(directory, lazy=True)
    
    print(set(clip.rec_equipment for clip in clips)) # print the list of recording equipments
