# Bearing-Fault-Diagnosis-Method-Based-on-Spectrum-Map-Information-Fusion-and-Convolutional-Neural-Network
Reproduce Published Article "Bearing Fault Diagnosis Method Based on Spectrum Map Information Fusion and Convolutional Neural Network""

This is a Python code implementation of the following publications: 

1.  Bearing-Fault-Diagnosis-Method-Based-on-Spectrum-Map-Information-Fusion-and-Convolutional-Neural-Network
- Authors: Baiyang Wang, Guifang Feng, Dongyue Huo, Yuyun Kang

- https://www.mdpi.com/2227-9717/10/7/1426/htm


2. Proposed Bearing Fault Diagnosis Method: 

2.1. VGG Convolutional Neural Network (CNN)
Current mainstream CNN models include AlexNet, GoogLeNet and the VGG network. Among them, the VGG network proposed by the Visual Geometry Group of Oxford University in 2014 is a neural network applied in image classification and recognition, with excellent feature extraction capabilities. The network contains 13 convolutional layers and 3 fully connected layers. VGG stacks multiple  3×3 convolution kernels to replace the large convolution kernels in traditional neural networks. Multiple convolution kernels effectively expand the number of channels. The pooling layer is used to reduce the width and height, making the constructed neural network more efficient, deeper, wider and less computationally intensive for large-scale neural networks [33]. The VGG network uses the ReLU function as the activation function. Unlike the Tanh and Sigmoid functions, the ReLU function is an unsaturated function, which means that it does not reduce the error of backpropagation, and the network converges faster, which can considerably reduce the training time. Based on these advantages, the fault diagnosis model uses the VGG network structure, which is shown in Figure 1.

![VGG network structure](https://user-images.githubusercontent.com/80536675/194580350-f470d7ac-0a00-45f3-ab2b-63e2e3dc78f6.png)

There was an excessive number of weight parameters of VGG-16, with three full-connection layer parameters accounting for a large proportion. The original parameter setting of VGG-16 was to complete 1000 classifications, with fewer signal classifications. Therefore, the first two fully connected layers only use half of the original number of nodes, namely 2048 nodes, and the third fully connected layer has 10 nodes corresponding to the classification category so as to improve the recognition accuracy and efficiency of the model.

2.2. Data Segmentation
The vibration signal of the bearing is a continuous 1D time series, so different data segmentation methods should be selected according to the signal type. The vibration signals should be sequentially and equally intercepted into different small segments, and the signal interval of each segment should be long enough to capture the local features of the signal. However, the number of sampling points in the original dataset is fixed; the more sampling points each sample contains, the fewer the samples. A smaller number of samples is not conducive to training of neural networks. Before the experiment, the influence of samples with different data length pairs on the results should be tested, and the optimal data segmentation length should be selected according to the model recognition accuracy [34,35]. The main dataset is the bearing vibration data from the CWRU dataset, using the single-channel drive-end (DE) accelerometer data.
For determination of data length, the data are not processed, and 2D images are drawn directly by the matplotlib function. The obtained datasets are shown in Table 1, and 9 signals of different lengths are shown in Figure 2. A total of 9 groups of datasets of different lengths are tested, i.e., 100, 300, 500, 700, 900, 1100, 1300, 1500 and 1700. The number of datasets constructed by 9 different sample lengths is the same, and the specific number is given in Table 1. The number of concrete can be divided into 10 categories: normal; inner race, 0.007 mils; ball, 0.007 mils; outer race, 0.007 mils; inner race, 0.014 mils; ball, 0.014 mils; outer race, 0.014 mils; inner race, 0.021 mils; ball, 0.021 mils; outer race, 0.021 mils.

![image](https://user-images.githubusercontent.com/80536675/194580467-3d6c5150-6b8b-48d7-b24b-154dd7339222.png)
Figure 2. The segmentation results of the same signal in the CWRU dataset are segmented according to the number of data points:(a) 100 points, (b) 300 points, (c) 500 points, (d) 700 points, (e) 900 points, (f) 1100 points, (g) 1300 points, (h) 1500 points and (i) 1700 points.

![image](https://user-images.githubusercontent.com/80536675/194580827-7441f5f0-14a4-4c8a-b441-a807bbbd2a06.png)


Nine single-channel datasets of different lengths were trained using convolutional neural network training. The data length of a single sample in all subsequent experiments presented in this paper was ultimately determined according to the training results of single-channel DE datasets and the accuracy of bearing fault diagnosis of the obtained model. Figure 3 shows the training results; the precision of training increases with an increased number of data points. However, when a single sample contains more than 900 data points, the precision of the model declines, and loss function values begin to change. Therefore, 900 data points were chosen as a sample.

![image](https://user-images.githubusercontent.com/80536675/194581147-0f1ab0d2-4140-469e-8c11-6fc0b0686fff.png)

After the length of each sample is determined, the data are divided according to the time series, as shown in Formula (1). The signal intervals do not overlap; x is the current time point, and n is the selected signal interval length.

T(x−n)≤T(x)≤T(x+n)    x∈(n,2n,…,xn)

A data segment after segmentation is shown in Figure 4. When 900 data points are divided into one sample, each sample is guaranteed to contain a cycle and comprehensive fault features.

![image](https://user-images.githubusercontent.com/80536675/194581221-79dfe7cb-1801-453c-a54f-98ea8ed510b0.png)

2.3. Spectral Analysis of Short-Time Fourier Transform (STFT)
Fourier transform can decompose a signal into several frequency components; each sinusoidal component has its own frequency and amplitude. Fourier transform can only determine which frequency components a signal contains for a period of time, but it cannot accurately determine the time when each frequency component appears. Therefore, it is possible to obtain similar spectrograms by analyzing signal fragments in different time domains. Therefore, Fourier transform is not suitable for signals with irregular periodic changes. The bearing vibration signal is a non-stationary signal containing different frequency components.
Therefore, it is not simple to use Fourier transform to analyze the spectrum of the signal. In order to avoid the loss of time information by Fourier transform of the entire sequence, local frequency parameters can be introduced, and Fourier transform can be used locally in the signal. By adding a window to intercept the segment of the signal, a window function (w(t)) is defined, as in Formula (2). The window function is moved to a certain center point (τ) and multiplied by the original signal to obtain the truncated signal (y(t)).

Then, Fourier transform is used to analyze the truncated signal (y(t)) and obtain the spectral distribution (X(ω)) of a segmented sequence according to Formula (3).

In real applications, because the signal is a discrete point sequence, the spectrum sequence (X[N]) is obtained. For the convenience of expression, we define the function  S(ω, τ) in Formula (4), which represents the spectral result (X(ω)) after transforming the original function when the center of the window function is τ  [36].


Corresponding to the discrete scene, S(ω,τ) is a two-dimensional matrix, and each column represents the result sequence of windowing the signal at different positions and performing Fourier transform on the obtained signal segment. After completing the Fourier transform operation of the first segment, the window function is moved to τ0, and the moving distance is generally less than the width of the window so as to ensure that there is a certain overlap between the two windows before and after, which we call overlap. The above operations are repeated, and the window is continuously slid to perform Fourier transform on the data truncated by the window to obtain the spectral results (S(ω,τ)) of all segments from τ0 to τN  [37,38], as shown in Figure 5.

![image](https://user-images.githubusercontent.com/80536675/194581413-a05fd3f7-9bc9-4d61-9631-e90eb4558b86.png)

The result of Fourier transform of each window is a complex two-dimensional matrix; each column of this matrix is the spectrum of a window, and the number of columns in the matrix is equal to the number of segments of the signal divided by the window. This is used to determine the magnitude of the complex number to obtain the real amplitude value; then, the color block is used to represent the amplitude of each column. The higher the amplitude, the brighter the color block, and the lower the amplitude, the darker the color block, the specific operation process is shown in Figure 6.

![image](https://user-images.githubusercontent.com/80536675/194581505-e9874d47-d6ef-45b7-88a5-646671a7af5e.png)

In this study, we used the pcolormesh() function in the matplotlib library to draw the spectrogram. The Hanning window is used as the window function. The Hanning window is suitable for non-periodic continuous signals to reduce the leakage phenomenon and improve the quality of the spectrum analysis. Formula (5) is the time domain expression for the length of the Hanning window. The length of the window function is set to 256, with an overlap of 50%. The time-domain signal is then divided into segments using a sliding window.

Figure 7 shows an effect diagram after coding. The STFT diagram is obtained when the upper part is divided into vibration data of a single channel and the lower part is divided into vibration data of two channels. The STFT matrices of the two channels are obtained and added together.

![image](https://user-images.githubusercontent.com/80536675/194581570-3b242234-4682-4a30-86a3-d603e4b2710a.png)
Figure 7. Short-time Fourier transform coding: (a) divided into vibration data of a single channel, (b) divided into vibration data of two channels.

2.4. Diagnostic Methods
The process of bearing fault diagnosis based on spectrum map information fusion and convolutional neural network proposed in this paper is shown in Figure 8. First, according to the intercept signal and signal segment of a certain length, an appropriate method is selected based on 1D vibration signal processing to convert a signal fragment to a 2D spectrum map. Then, the spectrum map dataset is divided into a training set and a validation set according to a certain proportion. The training set is input into VGG for training, and the validation set is input into the model to predict the fault type. The specific steps are as follows.


![image](https://user-images.githubusercontent.com/80536675/194581655-2ef5f813-77d8-43cb-bcaf-dfacb024b3ec.png)
Figure 8. Flow chart of rolling bearing fault diagnosis based on a spectrum map.

1. Sensors installed in different locations of the equipment collect vibration signals. In this study, we used collected vibration datasets rather than real-time vibration signals.
2. The collected vibration signals are processed, the appropriate length is selected as a sample and 1D data are processed by the STFT method. The processed 1D data are stored as 2D images by Matplotlib. When the dataset is multidimensional, a multichannel dataset is generated by data fusion to improve the recognition accuracy.
3. The spectrum map dataset is divided into a training set and a validation set according to a certain proportion.
4. Appropriate neural networks are selected for training. Finally, a VGG convolutional neural network is used to train the model on the training set to obtain the neural network prediction model of bearing faults.
5. The trained model is deployed to mechanical equipment for fault detection.

- I haven't use VGG model either, i've used Resnet18.
