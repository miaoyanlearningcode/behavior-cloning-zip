import os
import csv
import cv2
import numpy as np
import sklearn

samples = []
curPath = './data'
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)


from random import shuffle
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, curPath, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            angleOffset = [0.0, 0.5, -0.5]
            for batch_sample in batch_samples:
                for i in range(3):
                    name = curPath + '/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3]) + angleOffset[i]
                    
                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image,1))
                    angles.append(angle*(-1.0))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, curPath, batch_size=32)
validation_generator = generator(validation_samples, curPath, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
# model.add(Flatten(input_shape=(160,320,3)))
model.add(Convolution2D(4,3,3))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Convolution2D(8,3,3))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
exit()