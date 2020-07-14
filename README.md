

# Detecting Respiratory Disorders


## CS 7651 - Machine Learning (Team 7)


# Introduction


# Data

Data is taken from the _Respiratory Sound Database_ created by two research teams in Portugal and Greece. It consists of 920 recordings. Each recording is varying in length. A scatter plot of the length of recordings is given in **FIGURE**. Recordings were taken from 126 patients and each recording is annotated. Annotations are comprised of beginning and end times of each respiratory cycle and whether the cycle contains crackle and/or wheeze. Crackles and wheezes are called adventitious sounds and presence of them are used by health care professionals when diagnosing respiratory diseases. Number of respiratory cycles containing each adventitious cycle is shown in **PLOT**.

![](images/plt_clip_lengths.png)

![](images/plt_class_dist.png)



## Preprocessing

Preprocessing of the data starts from importing the sound files, resampling and cropping them. Since the recordings were taken by two different research teams with different recording devices, there are 3 different sampling rates (44100 Hz, 10000 Hz and 4000 Hz). All recordings were resampled to 44100 Hz and all clips are made 5 seconds long by zero padding shorter clips and cropping the longer ones.  `librosa` library was used in this project for reading the audio data and extracting features.

### Feature Extraction (MFCC)

Mel Frequency Cepstrum Coefficients were used as features of the sound clips. MFCCs are widely used in speech recognition systems. They are also being used extensively in previous work on detection of adventitious respiratory sounds, they provide a measure of short term power spectrum of time domain signals. Both the frequency and time content are important to distinguish between different adventitious sounds, since different adventitious sounds can exist in a single clip at different time periods and they differ in duration. Therefore, MFCC is helpful in capturing the change in frequency content of a signal over time. Frequencies are placed on a mel scale, which is a nonlinear scale of frequencies whose distances are percieved to be equal by the human auditory system. Output of MFCC is a 2 dimensional feature vector (time and frequency), which was then flattened into a one dimensional array before further processing.  

A sample output from MFCC content of clips containing different adventitious sounds is given below.

**FIGURE**

### Dataset Partitioning
Since the dataset does not include seperate recordings for training and testing, we randomly partition the dataset into training (80%) and testing (20%) by maintaining the class distribution for both sets. For the first classification method (SVM), we perform a 5-fold cross validation to pick the hyperparameters, therefore, no seperate validation dataset is required. As for the second classification method (CNN), we split the training dataset so that 70% of the original dataset is used for training and 10% is used for validation. Fig. **XX1** illustrates the class distribution for CNN system.

<p align="center">
<img src="images/classDistribution.png" width="600">
<br>
Figure **XX1**. Distribution of the dataset to be used for CNN-based system
</p>

# Classification Methods

## Principal Component Analysis (PCA) and Support Vector Machines (SVM) in Pipeline
As for the first classification method of our project, we combine an unsupervised learning method, PCA, with a supervised learning method, SVM. The main reason behind including PCA before SVM is to reduce the dimensionality of the dataset and hence increase the learning rate. The details for these methods are explained next.

###  Principal Component Analysis (PCA) for Dimensionality Reduction

The MFCC-based features exist in a 2-dimensional space where the first and second dimensions represent the time and frequency information, respectively. Hence, the <img src="https://render.githubusercontent.com/render/math?math=i^{th}">  input data <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}_i \in \mathbb{R}^{TxD}"> where T is the number of time-windows and D is the number of frequency bins. As can be seen in the figures above, the features are sparsely distributed and most of these features show similar characteristics across different classes. This implies that the dataset contains significant amount of redundant information. In an effort to reduce the dimensionality of the dataset and hence increase the learning rate while keeping the variation within the dataset, we propose to utilize principal component analysis (PCA).

PCA is a commonly used unsupervised learning technique in machine learning that projects the dataset into a lower dimensional dataset in an optimal manner that maximizes the remained variance <sup>[1](#pca)</sup>  . PCA performs a linear transformation and is therefore useful for datasets that are linearly seperable. In conventional PCA application, the input data for each sample is represented as a vector (1-dimensional). Therefore, the whole dataset can be represented as a 2-dimensional matrix that consists the stacked input vectors. Firstly, the dataset is centered to avoid a biased projection. Later, the centered data matrix is expressed in singular value decomposition form. The projection is performed by keeping the largest singular values and their corresponding eigenvectors. The number of the singular values that are utilized can be determined manually or can be chosen so that the explained variance of the original dataset achieves a certain threshold.

Our MFCC-based features lie in a 2-dimensional space. To be able to utilize the conventional PCA scheme, we flatten the features so that the features are represented as a vector. Then, we can apply the common PCA procedure. However, PCA is not agnostic to different scalings of the features. Therefore, we standardize the data so that all features are similarly scaled.

To give a perspective of the dimensions, when the maximum length of the recording are limited to 5 seconds, the resulting MFCC features have the dimension 20 x 431. Therefore, we have 8620 features in total. As explained above, the values for most of these features are the same across the different classes and redundant. In Figure **XX2**, how the explained cumulative variance changes for increasing number of components is presented. We note that we still keep the 99% of the original variance when the dimensionality is reduced to 1916. This reduction is very significant because it becomes useful to increase the learning rate in the next step.

<p align="center">
<img src="images/PCA_explained_varience.png" width="600">
<br>
Figure **XX2**. Explained variance for increasing number of kept principal components
</p>

### Support Vector Machines (SVM)
## Convolutional Neural Networks (CNN)

As the second classification approach, we propose to use a Convolutional Network Network based system. The Convolutional Network Network (CNN) is a neural network classification technique that is commonly used in image classification <sup>[2](#imagenet)</sup> <sup>[3](#vggnet)</sup>. As opposed to the traditional neural networks, where each input feature is associated with seperate parameters, in CNN, parameters are shared among the features. This allows the network for learning local features. By this means, CNN automatically learns the important features without requiring extra feature extraction.
CNN-based architectures construct a deep layered structure through convolutional kernels, which are learned from the data to extract complex features. Furthermore, CNN is computationally efficient. Convolution and pooling operations allow for parameter sharing and efficient computation.

In our project, the MFCC-based features are 2-dimensional. Therefore, they can be treated as images and the assignment can be translated into an image classification task. After experimenting with commonly used CNN structures such as AlexNet <sup>[2](#imagenet)</sup> and VGGNet <sup>[3](#vggnet)</sup>, we designed our own CNN-based neural network structure as shown in Fig. **??**.

Our network includes three convolutional layers (each followed by a max-pooling layer) and four fully connected layers as well as the output layer. The convolution operations are performed with a kernel size of 15x15 and stride of 1. The fully connected layers have 6784, 2048, 1024 and 128 neurons, respectively. The activation function for all convolutional and fully connected layers is Rectified Linear Unit (ReLU). The output layer, consisting of 4 nodes, implements a softmax activation function. The max-pooling operations are performed with a kernel of size 2x2 and stride 1.

<p align="center">
<img src="images/cnn_2d_arch.png" width="1000">
<br>
Figure XX. Architecture of the CNN-based neural network
</p>


The proposed neural network system above consists of over 16 million parameters to be trained. Considering the dataset size, this is a significantly large number of parameters. To increase the training speed, we use Adam optimizer <sup>[4](#adam)</sup>. We specify our loss function as categorical cross entropy
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=L(y,\hat{y})=-\sum_{i=1}^{C} y_i\log(f(\hat{y}_i))">.</p>
where,
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=f(\hat{y}_i)=\frac{e^{\hat{y}_i}}{\sum_{j=1}^{C}e^{\hat{y}_j}}">,</p>
and the number of classes is C = 4. Then, we train our algorithm and evaluate it on the validation set to choose the the number of epochs.


# Evaluation & Results
## SVM Results

Our best SVM model achieved an accuracy of 69%. Interestingly, the recall percentages correlate well with the distribution of classes in our data. When looking at the unbalanced dataset, as less training data was available in each class, the corresponding recall values also decreased. Figure 0 is the confusion matrix with percent recall values, and figure 1 illustrates this by normalizing the number of clips in each class and the recall of each class.

<p align="center">
<img src="images/eval_fig0.png" width="500">
<br>
Figure 0. Normalized confusion matrix for SVM model
</p>

<p align="center">
<img src="images/eval_fig1.png" width="600">
<br>
Figure 1. Comparison of normalized class distribution and normalized recall for each class in SVM model
</p>

The unbalanced data could be the reason for our relatively low accuracy of 69%. The healthy class, which had the most data available (3642 clips) achieved a recall of 82%, while the both class, with the least data available (506 clips) achieved a recall of 37%.

## CNN Results

Our best CNN model achieved an accuracy of 71%. The normalized confusion matrix is shown in Figure 2.

<p align="center">
<img src="images/eval_fig2.png" width="500">
<br>
Figure 2. Normalized confusion matrix for CNN model
</p>

Overfitting starts to happen at around the 20th epoch. After the 20th epoch, the testing accuracy starts to increase at a noticeably slower rate than the training accuracy. The testing loss also stops decreasing at this point, while the training loss continues to decrease. Although more training at each epoch does result in a higher validation accuracy, the accuracy gain is much less when compared to the training accuracy. A graph of the training and validation accuracy is shown in Figure 3, and a graph of the training and validation loss is shown in Figure 4.

<p align="center">
<img src="images/eval_fig3.png" width="600">
<br>
Figure 3. Training and validation accuracy for CNN model across 30 epochs
</p>

<p align="center">
<img src="images/eval_fig4.png" width="600">
<br>
Figure 4. Training and validation loss for CNN model across 30 epochs
</p>

Like the SVM model, the recall percentages for the CNN model also correlate well with the distribution of classes in our data. The graph of the normalized class distribution and recall comparison is shown in Figure 5.

<p align="center">
<img src="images/eval_fig5.png" width="600">
<br>
Figure 5. Comparison of normalized class distribution and normalized recall for each class in CNN model
</p>

## Dataset Evaluation
The dataset itself was a difficult dataset to work with. Aside from the unbalanced part of it that was discussed previously, there were various other features that could affect our accuracies.

One aspect of the data that likely reduced our accuracy was the format of the data itself. All the clips were of different lengths, ranging from 0.2 to 16.2 seconds. The clips were also not sampled at the same sampling rate. This required us to augment the data through zero-padding, cropping, filtering, and up-sampling or down-sampling, which removed from the truth of the actual data and could cause problems in the training process.

Another aspect of the data that could have reduced our accuracy was how the data was gathered. Across all the clips, there were four different recording devices used, two different acquisition modes, and six different locations of the chest that were recorded. Our models did not account for any of these differences.

Furthermore, considering the number of parameters to be trained (over 16 million) in the CNN implementation, the dataset size is very small (4827 total samples) and this restricts the learning capability of the network. 
# Discussion & Conclusion

For the CNN structure, the accuracy results turned out to be comparable for different kernel sizes (3x3, 5x5, and 11x11), therefore, we only report the results for the best performing kernel size (15x15). **-->Can also be mentioned somewhere in the results**

In addition to the 2 dimensional CNN, we tried to use 1 dimensional CNN. For that, we used two different input types: 1) Flattened MFCC coefficients of size 8620x1, 2) Features obtained after applying PCA (1916x1). Training the former network took significantly long amount of time (300 s/epoch) since it required training 71 million parameters. The highest accuracy achieved with such a structure was 63%. On the other hand, training the second network took considerably less time (70 s/epoch) at the expense of significantly lower accuracy (54%). These results indicate that 2 dimensional CNN structure outperforms 1 dimensional CNN structures for this dataset. **-->Can also be moved to the results**

Possible considerations to increase the performance of the system are listed as follows:
* Using a larger dataset that has a balanced class distribution.
* Utilizing other feature extraction methods such as short-time Fourier transform (STFT)
* Applying advanced signal processing techniques to extract more informative and distinctive features from the recordings.
* ??

# References
<a name="pca">[1]</a>: Wold, Svante, Kim Esbensen, and Paul Geladi. "Principal component analysis." Chemometrics and intelligent laboratory systems 2.1-3 (1987): 37-52.
<a name="imagenet">[2]</a>: Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
<a name="vggnet">[3]</a>: Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
<a name="adam">[4]</a>: Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
