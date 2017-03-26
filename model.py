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


correction = 0.15
nb_epoch = 10
dataSets = [
            "./dataSet/1_anti_normal_1/",
            "./dataSet/1_clock_normal_1/",
            "./dataSet/1_clock_normal_2/",
            "./dataSet/1_anti_normal_2/",
            
            "./dataSet/1_clock_recover_3/",
#            "./dataSet/1_clock_recover_1/",
#            "./dataSet/1_anti_recover_3/",
            
###            "./dataSet/1_clock_recover_2/",
###            "./dataSet/1_anti_recover_2/",
            
#            "./dataSet/1_clock_curves_1/",
##            "./dataSet/1_anti_curves_1/",
#            
#            "./dataSet/2_clock_normal_1/",
#            "./dataSet/2_anti_normal_1/",  
#            "./dataSet/2_clock_normal_2/",
#            "./dataSet/2_anti_normal_2/"     
            ]

samples = []

def addFiles(folderPath):  
    with open(folderPath+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line+[folderPath])



for path in dataSets:
    addFiles(path)
        

def prepareSamples(samples, training = True):
    result = []         # in (imgPath, angleValue format, flip )
    for row in samples:
        
        center_name = row[-1]+'IMG/'+row[0].split('/')[-1]  
        left_name = row[-1]+'IMG/'+row[1].split('/')[-1]
        right_name = row[-1]+'IMG/'+row[2].split('/')[-1]
        center_angle = float(row[3])
        
        result.append((center_name, center_angle, False))
        if training:
            result.append((left_name, center_angle + correction, False))
            result.append((right_name, center_angle - correction, False))
            result.append((center_name, -1*center_angle, True))
#            result.append((left_name, -1*center_angle + correction, True))
#            result.append((right_name, -1*center_angle - correction, True))
    #shuffle data
    return sklearn.utils.shuffle(result)


def dataGenerator(samples, batch_size=32, training=True):    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for row in batch_samples:
                img = cv2.imread(row[0])   
                if row[2]:
                    img=cv2.flip(img,1)
                angle = row[1]
                images.append(img)
                angles.append(angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_samples = prepareSamples(train_samples)
validation_samples = prepareSamples(validation_samples, training=False)

# compile and train the model using the generator function
train_generator = dataGenerator(train_samples, batch_size=50, training=True)
validation_generator = dataGenerator(validation_samples, batch_size=50, training=False)

col, row, ch = 160, 320, 3  # Trimmed image format

model = Sequential()
with tf.name_scope('normalize_1'):
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(col, row, ch), output_shape=(col, row, ch), name="cut__img" ))

with tf.name_scope('Cropping2D_1'):
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(col, row, ch)))
    print(model.output_shape)

with tf.name_scope('conv1'):
    model.add(Convolution2D(24, 5, 5, border_mode='valid' ))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    print(model.output_shape)
    
with tf.name_scope('conv2'):
    model.add(Convolution2D(36, 5, 5, border_mode='valid' ))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    print(model.output_shape)
    
with tf.name_scope('conv3'):
    model.add(Convolution2D(36, 5, 5, border_mode='valid' ))
    model.add(Activation('relu'))
    print(model.output_shape)
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
    print(model.output_shape)
    
with tf.name_scope('fc1'):
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
with tf.name_scope('fc2'):
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
with tf.name_scope('fc3'):
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
with tf.name_scope('fc4'):
    model.add(Dense(10))
with tf.name_scope('fc5'):
    model.add(Dense(1))

tfBoard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=2, write_graph=True)

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch= len(train_samples), 
                                     validation_data=validation_generator, 
                                     callbacks = [tfBoard],
                                     nb_val_samples=len(validation_samples), nb_epoch=nb_epoch)

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
