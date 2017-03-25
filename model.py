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


correction = 0.2
dataSets = [
#"./dataSet/1_anti_normal_1/",
#            "./dataSet/1_clock_normal_1/",
##            "./dataSet/1_clock_normal_2/" ## have problem
#            "./dataSet/1_anti_normal_2/",
#            "./dataSet/1_clock_recover_1/",
#            "./dataSet/1_anti_recover_1/",
            "./dataSet/2_clock_normal_1/",
            "./dataSet/2_anti_normal_1/",  
            "./dataSet/2_clock_normal_2/",
            "./dataSet/2_anti_normal_2/",            
            ]

samples = []

def addFiles(folderPath):  
    with open(folderPath+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line+[folderPath])



for path in dataSets:
    addFiles(path)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#shuffle data
sklearn.utils.shuffle(train_samples)

def generator(samples, batch_size=32, training=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
#        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = batch_sample[-1]+'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)                
                center_angle = float(batch_sample[3])
                
                images.append(center_image)
                angles.append(center_angle)
                
                if training:
                    
                    left_name = batch_sample[-1]+'IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(left_name)                
                    left_angle = float(batch_sample[3])+ correction
                    
                    right_name = batch_sample[-1]+'IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(right_name)                
                    right_angle = float(batch_sample[3])- correction
                    
        
                    
                    images.append(left_image)
                    angles.append(left_angle)
                    
                    images.append(right_image)
                    angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, training=True)
validation_generator = generator(validation_samples, batch_size=32, training=False)

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
                                     samples_per_epoch= len(train_samples*3), 
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

plt.figure()
plt.plot(history_object.history['acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.show()


model.save('model.h5')
