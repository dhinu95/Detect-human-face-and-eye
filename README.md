# EMOTION RECOGNITION.
## Built a model to recognize emotions from human face using Neural Network (CNN). 

### Process
#### 1. Dataset (FER2013.csv)
   In this project used FER2013 dataset and created own face dataset using Cascade Classifier / Violes-Jones method.Collective of 36,210      samples.

#### 2. Preprocessing Dataset (Preprocessing.py)
   In FER2013 dataset there are totally 7 emotions and in own face dataset 6 emotions. So, removed 'disgust' emotion from FER2013.Finally    having 35,720 samples
   
#### 3. Train & Test (Train_test.py)
   In Test and Train, split dataset into 90:10 for Training and Testing respectively.
   
#### 4. Keras Model (Keras_model.py)
   In keras model, use Convolutional Neural Network (CNN) in which it can take in an input image, assign importance (learnable weights and    biases) to various aspects/objects in the image and be able to differentiate one from the other. Tried different module of different      filter size to attain maximum accuracy.
## Finally got accuract of 86.4336%
