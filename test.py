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
name = curPath + '/IMG/'+samples[1][0].split('/')[-1]
print (name)
center_image = cv2.imread(name)
print(center_image)

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
            for batch_sample in batch_samples:
                name = curPath + '/IMG/'+batch_sample[0].split('/')[-1]
                # print(name)	
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, curPath, batch_size=32)
validation_generator = generator(validation_samples, curPath, batch_size=32)

print(train_generator.shape)
