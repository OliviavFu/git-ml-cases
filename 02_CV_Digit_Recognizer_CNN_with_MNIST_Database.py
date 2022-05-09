# Kaggle Data Source: https://www.kaggle.com/datasets/ipythonx/mnist-in-cv

# The MNIST database (Modified National Institute of Standards and Technology database) is a large database 
# of handwritten digits that is commonly used for training various image processing systems.

# We will build a Digit Recognizer using Convolutional neural networks. This is a multiclass classification 
# problem and two neural nets are compared: one with BatchNormalization, another without.   


#################################################################################################################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Lambda, Flatten, Dropout, BatchNormalization, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
# from keras.optimizers import adam_v2

# Load the data
df_train = pd.read_csv('/kaggle/input/mnist-in-cv/mnist_classification/train.csv')
df_test = pd.read_csv('/kaggle/input/mnist-in-cv/mnist_classification/test.csv')

# Get X, y 
X_train = df_train.iloc[:,1:].values.astype('float32').reshape(X_train.shape[0],28,28,1) # reshape for CNN
y_train = df_train.iloc[:,0].values.astype('int32')
X_test = df_test.values.astype('float32').reshape(X_test.shape[0],28,28,1)

# Side note: In order to work with a GPU, we need to change all data types to 32-bit floats; y works with 32-bit int. 

# Prepare for feature normalization 
X_mean = X_train.mean().astype('float32')
X_std = X_train.std().astype('float32')
def normalize(x):
    return (x - X_mean)/(X_std + 10**(-10))

# OneHotEncode label category for training 
y_train = to_categorical(y_train)

# Define an early stop rule
callback = EarlyStopping(monitor='loss',patience=3)

# Design neural network architecture: model WITHOUT BatchNorm
model = Sequential([
    Lambda(normalize, input_shape=(28,28,1)),
    Convolution2D(32, (3,3), activation = 'relu'),
    Convolution2D(32, (3,3), activation = 'relu'),
    MaxPooling2D(),
    Convolution2D(64, (3,3), activation = 'relu'),
    Convolution2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation = 'relu'), 
    Dropout(0.2),
    Dense(256, activation = 'relu'),
    Dropout(0.2),
    Dense(128, activation = 'relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation = 'softmax')
])

# Side note: The higher the number of filters in convolution layer, the higher the number of abstractions that your 
# Network is able to extract from image data. The reason why the number of filters is generally ascending is that at 
# the input layer the Network receives raw pixel data. Raw data are always noisy, and this is especially true for 
# image data.

# Train the model
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks = [callback], verbose = 2)

# Make predictions and randomly check some sample results 
preds = model.predict(X_test, verbose = 0)
preds_cat_2 = np.argmax(preds.astype('int32'), axis = 1) # label category reverts from one hot encoder. 

X_test_view = X_test.reshape(X_test.shape[0],28,28)
for i in range(0,9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_test_view[i+60])
    plt.title(preds_cat_2[i+60])
    
# Design neural network architecture: model WITH BatchNorm
model_bn = Sequential([
    Lambda(normalize, input_shape=(28,28,1)),
    Convolution2D(32, (3,3), activation = 'relu'),
    BatchNormalization(axis = 1),
    Convolution2D(32, (3,3), activation = 'relu'),
    MaxPooling2D(),
    BatchNormalization(axis = 1),
    Convolution2D(64, (3,3), activation = 'relu'),
    BatchNormalization(axis = 1),
    Convolution2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    BatchNormalization(),
    Dense(512, activation = 'relu'), 
    Dropout(0.2),
    BatchNormalization(),
    Dense(256, activation = 'relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(128, activation = 'relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation = 'softmax')
])

# Train the model
model_bn.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_bn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks = [callback], verbose = 2)

# Make predictions and randomly check some sample results 
preds_bn = model_bn.predict(X_test, verbose = 0)
preds_cat = np.argmax(preds_bn.astype('int32'), axis = 1) # label category reverts from one hot encoder. 

X_test_view = X_test.reshape(X_test.shape[0],28,28)
for i in range(0,9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_test_view[i+10])
    plt.title(preds_cat[i+10])

# model WITHOUT BatchNorm performance: validation accuracy starts from 0.9721 in the 1st epoch to 0.9929 in the end.
# model WITH BatchNorm performance: validation accuracy starts from 0.9950 in the 1st epoch to 0.9936 in the end.

# Adam optimizer (adaptive learning rate + momentum) is used for both networks, we can see that the final model 
# performance ends up very close for this simple dataset. BatchNorm could be helpful to achieve better performance 
# with less epoches and maybe speed up the training process for complex CNN jobs. In the meanwhile, regular Adam 
# without BatchNorm could work fine for many cases too. 

# Side note 1: Some data aumentation techniques for small datasets: 
    # Cropping
    # Rotating
    # Scaling
    # Translating
    # Flipping
    # Adding Gaussian noise to input images etc.

# Side note 2: Data augmentation is applied whilst training to make the model robust. Whereas, adversarial 
# attacks (carefully tailored) are applied to images which are then sent through a trained model to check its 
# robustness and security. 



