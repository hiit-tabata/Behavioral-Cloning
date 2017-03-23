# This file is to train the model
# python drive.py model.h5

# Here are some general guidelines for data collection:
#
# two or three laps of center lane driving
# one lap of recovery driving from the sides
# one lap focusing on driving smoothly around curves

# video.py usage
#python video.py run1 --fps 48


import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import keras
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout
import tensorflow as tf

samples = []
with open('./dataSet/track1_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('./dataSet/track1_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#TODO shuffle data

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
#        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './dataSet/track1_1/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

col, row, ch = 160, 320, 3  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(col, row, ch), output_shape=(col, row, ch), name="cut__img" ))
model.add(Cropping2D(cropping=((75,20), (0,0)), input_shape=(col, row, ch)))
print(model.output_shape)

with tf.name_scope('conv1'):
    model.add(Convolution2D(24, 5, 5, border_mode='valid' ))
    model.add(Activation('relu'))
    print(model.output_shape)
    
with tf.name_scope('conv2'):
    model.add(Convolution2D(36, 5, 5, border_mode='valid' ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    print(model.output_shape)
    
with tf.name_scope('conv3'):
    model.add(Convolution2D(36, 5, 5, border_mode='valid' ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    print(model.output_shape)
    
with tf.name_scope('conv4'):
    model.add(Convolution2D(48, 3, 3, border_mode='valid' ))
    model.add(Activation('relu'))
    print(model.output_shape)
    
with tf.name_scope('conv5'):
    model.add(Convolution2D(64, 3, 3, border_mode='valid' ))
    model.add(Activation('relu'))
    print(model.output_shape)
    
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dense(1))

tfBoard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=2, write_graph=True)

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch= len(train_samples), 
                                     validation_data=validation_generator, 
                                     callbacks = [tfBoard],
                                     nb_val_samples=len(validation_samples), nb_epoch=10)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


model.save('model.h5')
