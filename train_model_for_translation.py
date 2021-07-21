import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

classes = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

train_dir = "C:/code/gesture/asl/train/"
test_dir = "C:/code/gesture/asl/test/"

for classe in classes :
    path = os.path.join (train_dir, classe)
    for image in os.listdir (path):
        image_arr = cv2.imread (os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
        print (image_arr)
        plt.imshow(image_arr , cmap = 'gray')
        plt.show ()
        break
    break

IMG_SIZE = 40
image_resize_arr = cv2.resize(image_arr,(IMG_SIZE,IMG_SIZE))
plt.imshow (image_resize_arr, cmap = 'gray')
plt.show()

training_data = []

def create_training_data ():
    for classe in classes :
        
        path = train_dir + "/" + classe
        for img in os.listdir(path):
            if ord (classe) >= 97:
                img_array = cv2.imread (os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize (img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append ([new_array, ord(classe)-87])
            else:
                img_array = cv2.imread (os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize (img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append ([new_array, int(classe)])
            
             
create_training_data()
print (len(training_data))
print (np.shape(training_data))

import random

random.shuffle(training_data)

X_train = []
y_train = []

for features, label in training_data :
    X_train.append(features)
    y_train.append(label)
    

X_train = np.array (X_train).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
print (X_train.shape)
print (y_train[:20])

testing_data = []

def create_testing_data ():
    for classe in classes :
        path = test_dir + "/" + classe
        for img in os.listdir(path):
            if ord (classe) >= 97:
                img_array = cv2.imread (os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize (img_array, (IMG_SIZE,IMG_SIZE))
                testing_data.append ([new_array, ord(classe)-87])
            else :
                img_array = cv2.imread (os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize (img_array, (IMG_SIZE,IMG_SIZE))
                testing_data.append ([new_array, int(classe)])
            
             
create_testing_data ()
print (len(testing_data))

random.shuffle(testing_data)

X_test = []
y_test = []

for features, label in testing_data :
    X_test.append(features)
    y_test.append(label)
    

X_test = np.array (X_test).reshape(-1,IMG_SIZE, IMG_SIZE, 1)

from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape (X_train.shape[0], *(40,40,1))
X_test = X_test.reshape (X_test.shape[0], *(40,40,1))

print (X_test.shape)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, dtype = 'string') y_test = to_categorical(y_test)

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D,MaxPooling2D
import pickle
from tensorflow.keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (40,40,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(36, activation = "softmax"))


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()

epochs = 30  # for better result increase the epochs
batch_size = 128

history = model.fit(X_train,y_train, batch_size=batch_size, epochs = epochs, verbose=1 , validation_data = (X_test,y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save("model_cnn.h5")

