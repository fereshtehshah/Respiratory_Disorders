This file contains documentation of the code for importing the dataset into memory. The first part contains information about the file structure of the dataset, second section contains detailed descriptions of the functions and classes and third part contains examples for how to use it.

# File Structure of the Dataset

File names of the recordings are in the following format: `101_1b1_Al_sc_AKGC417L`. The information included in the file name is described in the following table

| Patient number         | Recording index                        | Chest location                                                                                                            | Acquisition mode                              | Recording equipment                                         |
| ---------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------- |
| `101`, `102` ... `206` | *What does recording index represent?* | `Tc` (Trachea)<br/>`Al`(Anterior left)<br/>`Ar` (Anterior right)<br/>`Pl`(Posterior left)<br/>`Pr` (Posterior right)<br/> | `sc` (single channel) or `mc` (multi channel) | `AKGC417L`<br/>`LittC2SE`<br/>`Litt3200`<br/>`Meditron`|

Each recording is accompanied by a `txt` file with the same name in which each line contains four columns. These columns are

1. Beginning of respiratory cycle

2. End of respiratory cycle

3. Presence/absence of crackles (presence=1, absence=0)

4. Presence/absence of wheezes (presence=1, absence=0)

# Function and Class Reference

## `import_all_files(directory, sr=None, lazy=False)`

This function reads all sound and annotation files in the given `directory` and returns a 2D list of `Clip` objects for each cycle in each recording. If there are `n` recordings in the given directory, the output of this function will be a list of `n` lists  where each list contains `Clip`s of the corresponding recording. Sound files are resampled to the sampling rate given by `sr`. Original sampling rate is preserved if `sr` is not provided. Argument `lazy` determines whether all sound files are read into the memory at the time of creation of each clip object. When the argument `lazy` is `False`, which is the default value, all sound clips are read and loaded into the memory of `Clip` objects. If `lazy` is `True`,  sound recordings are not loaded into `Clip` objects but all other annotations are. In that case sound files are loaded the first time the user tries to access the sound data. This might be useful in cases when only some of the clips are going to be used and the user doesn't want to spend time and memory to load all sound data but still needs to have all clip objects with annotations so as to choose which ones to load.



## `class Clip`

Thie class represents each clip in the sound recording. Each clip contains a crackle and/or a wheeze or neither.  Information contained in the `Clip` object is listed below:

* `patient_id` is an `int` representing the patient id

* `rec_i` is a `str` representing the recording index

* `chest_loc` is a `str` representing the chest location 

* `acq_mode` is a `str` representing the acquisition mode (`sc` or `mc`)

* `rec_equipment` is a `str` representing the recording equipment

* `crackle` is a `bool` value representing presence of crackle in the clip

* `wheeze` is a `bool` value representing presence of wheeze in the clip

* `start_t` is a `float` that shows the starting time of the clip

* `end_t` is a `float that shows the end time of the clip

* `sr` is an `int` representing the sampling rate in Hz

* `sound_data` is a `numpy.array` of `float32` representing the sound data of the clip

* `recording` is an object which represents the underlying sound file that the clip is taken from. It doesn't need to be accessed but it is used for the lazy loading situation

Functions defined in `Clip` are utility functions used for generating clips from the the recording and annotation files. 



### `Clip.generate_from_file(filename, sr=None, lazy=False)`

This function can be used for generating clip objects from the given file location in `filename`. If `sr` is provided, the sound file will be resampled to the given value, otherwise original sampling rate is preserved.  If`lazy`  is `True`,  the sound file is not read into the memory at the time of creating the clip object, but all other information about the clip is saved. Sound data is loaded the first time the user tries to access `sound_data` of the object. 



### `clip.recording_name()`

Returns the name of the recording that the clip is taken from.

# Examples

Example for loading all files into the memory using `import_all_files()` function:

```python
>>> directory = "C:\\Users\\Serhat\\OneDrive - Georgia Institute of Technology\\Classes\\CS 7641\\project\\110374_267422_bundle_archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files"
>>> clips = import_all_files(directory) # flat array containing all clips
Files to Clips: 100%|██████████| 920/920 [01:40<00:00,  9.15it/s]
100%|██████████| 920/920 [00:00<00:00, 592743.42it/s]
```

Each member of the `clips` array will be a `Clip`

```python
>>> clips[0]
Clip(101_1b1_Al_sc_Meditron, c=0, w=0, (0.036s, 0.579s))

>>> clips[0].sound_data
array([-0.0486145 , -0.04852295, -0.04833984, ..., -0.0869751 ,
       -0.08694458, -0.08709717], dtype=float32

>>> print(clips[0]) # prints general information about the clip
Clip(101_1b1_Al_sc_Meditron, c=0, w=0, (0.036s, 0.579s))

# original sampling rate will be preserved since not sr parameter was 
# given is not provided when importing
>>> clips[0].sr 
44100
```

Clips can be grouped based on specific criteria

```python
# get clips that contain only wheeze
>>> [clip for clip in clips if (clip.wheeze and not clip.crackle)]
[Clip(103_2b2_Ar_mc_LittC2SE, c=0, w=1, (0.364s, 3.250s)),
 Clip(103_2b2_Ar_mc_LittC2SE, c=0, w=1, (6.636s, 11.179s)),
 Clip(103_2b2_Ar_mc_LittC2SE, c=0, w=1, (11.179s, 14.250s)),
 Clip(103_2b2_Ar_mc_LittC2SE, c=0, w=1, (14.250s, 16.993s)),
 Clip(104_1b1_Ar_sc_Litt3200, c=0, w=1, (0.000s, 0.545s)),
...
]

# get a list of sound_data from a specific device
>>> [clips.sound_data for clip in clips if clip.rec_equipment=="LittC2SE"]
[array([0.11270142, 0.1121521 , 0.11227417, ..., 0.01907349, 0.01864624,
        0.01922607], dtype=float32),
 array([0.01950073, 0.01977539, 0.01947021, ..., 0.5267029 , 0.5274353 ,
        0.52786255], dtype=float32),
 ...
 array([0.00769043, 0.00784302, 0.00808716, ..., 0.16845703, 0.16833496,
        0.1685791 ], dtype=float32),
 array([ 0.1685791 ,  0.16851807,  0.16860962, ..., -0.06219482,
        -0.06234741, -0.06213379], dtype=float32)]
```

As it can be seen from the examples, there might be cases where only a small group of clips might be required. In those cases, it might be useful to use lazy importing of the clips so as to prevent waiting for too long for all clips to be loaded and to save memory.

```python
directory = "C:\\Users\\Serhat\\OneDrive - Georgia Institute of Technology\\Classes\\CS 7641\\project\\110374_267422_bundle_archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files"
clips = import_all_files(directory, lazy=True)  # Runs faster since sound files are not loaded
Files to Clips: 100%|██████████| 920/920 [00:06<00:00, 148.91it/s]
100%|██████████| 920/920 [00:00<00:00, 543104.81it/s]
```

All metada about clips are still accesible

```python
>>> clips[0]
Clip(101_1b1_Al_sc_Meditron, c=0, w=0, (0.036s, 0.579s))

# This line forces sound data for clips with patient id 101 to be loaded
# other clips are not loaded until requested
>>> [clips.sound_data for clip in clips if clip.patient_id==101]
[array([-0.0486145 , -0.04852295, -0.04833984, ..., -0.0869751 ,
        -0.08694458, -0.08709717], dtype=float32),
 array([-0.08721924, -0.08724976, -0.08721924, ..., -0.02172852,
        -0.02172852, -0.02178955], dtype=float32),
...
array([ 0.06091309,  0.06100464,  0.06100464, ..., -0.00296021,
        -0.00308228, -0.0032959 ], dtype=float32),
 array([-0.00332642, -0.00363159, -0.00360107, ..., -0.08328247,
        -0.08331299, -0.08352661], dtype=float32)]
```
