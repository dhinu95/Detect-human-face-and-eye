# EMOTION RECOGNITION.
### Built a model to recognize emotions from human face using Neural Network(CNN). 

## Libraries used:
#### Keras
   Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive 
   Toolkit, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, 
   modular, and extensible.
   
#### Pandas
   Pandas is the most popular python library that is used for data manipulation and data analysis.n particular, it offers data structures 
   and operations for manipulating numerical tables and time series. 

#### Numpy
   NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for 
   working with these arrays.
   
   

## Process
#### 1. Dataset (FER2013.csv)
   In this project used FER2013 dataset and created own face dataset using Cascade Classifier / Violes-Jones method.Collective of 36,210      samples.

#### 2. Preprocessing Dataset (Preprocessing.py)
   In FER2013 dataset there are totally 7 emotions and in own face dataset 6 emotions. So, removed 'disgust' emotion from FER2013.Finally    having 35,720 samples
   
#### 3. Train & Test (Train_test.py)
   In Test and Train, split dataset into 90:10 for Training and Testing respectively.
   
#### 4. Keras Model (Keras_model.py)
   In keras model, use Convolutional Neural Network (CNN) in which it can take in an input image, assign importance (learnable weights and    biases) to various aspects/objects in the image and be able to differentiate one from the other. Tried different module of different      filter size to attain maximum accuracy.
## Finally got Accuracy of 86.4332%
