from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import model_from_json
import os
import cv2
import numpy as np
from pathlib import Path
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR )
import random


class CNN:

    def __init__(self):
        self.letter_to_value = {}
        self.value_to_letter = {}
        count = 0
        self.list = []
        for i in xrange(65, 91):
            self.letter_to_value[chr(i)] = count
            self.value_to_letter[count] = chr(i)
            count += 1
        self.letter_to_value['nothing'] = 27
        self.letter_to_value['space'] = 28
        self.value_to_letter[27] = 'nothing'
        self.value_to_letter[28] = 'space'

    def fetch_data(self, dir_path):
        images = []
        entries = os.listdir(dir_path)
        for entry in entries:
            images.append(entry)
        return images

    def data_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (200, 200))
        frame = np.array(frame)
        frame = frame.reshape((1, 200, 200, 1))
        frame = frame/255.0
        return frame

    def prepare_test_data(self, images):
        dir_path = '/home/oem/Desktop/asl_alphabet_test/'
        inputs = []
        labels = []
        for image in images:
            img = cv2.imread(dir_path + image, cv2.IMREAD_GRAYSCALE)
            inputs.append(img)
            labels.append(image[:1])
        inputs= np.array(inputs)
        inputs = inputs.reshape((len(inputs), 200, 200, 1))
        inputs = inputs/255.0
        return inputs, labels

    def prepare_data(self, images, letter):
        dir_path = '/home/oem/Desktop/asl_alphabet_train/' + letter +'/'
        for image in images[:200]:
            #Normal
            img = cv2.imread(dir_path + image, cv2.IMREAD_GRAYSCALE)
            self.list.append([img, self.letter_to_value[letter]])
            #Invert
            img = cv2.bitwise_not(img)
            self.list.append([img, self.letter_to_value[letter]])

    def mix_data(self):
        random.seed(10)
        random.shuffle(self.list)
        inputs = []
        labels = []
        for item in self.list:
            inputs.append(item[0])
            labels.append(item[1])
        inputs= np.array(inputs)
        inputs = inputs.reshape((len(inputs), 200, 200, 1))
        inputs = inputs/255.0
        labels = np.array(labels)
        labels = labels.reshape((len(labels), 1))
        return inputs, labels

    def save_model(self, model):
        model.save('model2.h5')
        print ('Saved Model To Disk')
        print ('--------------------')

    def load_model(self):
        model = models.load_model('model2.h5', custom_objects={'auc': self.auc})
        print ('Loaded Model From Disk')
        print ('--------------------')
        return model

    def create_cnn_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(28, activation='softmax'))
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        print ('Created A New Model')
        print ('--------------------')
        return model

    def train_cnn_model(self, model, train_images, train_labels):
        model.fit(train_images, train_labels, epochs=10, batch_size=150)

    def predict_sample(self, model, test_images, test_labels):
        print ('Predicting')
        print ('--------------------')
        prediction = model.predict_classes(test_images)
        i = 0
        for p in prediction:
            if p in self.value_to_letter:
                print ('Result:  ' + self.value_to_letter[p] + ', Actual:  ' + test_labels[i])
            i += 1

    def predict_frame(self, model, frame):
        print ('Predicting Frame')
        print ('--------------------')

        prediction = model.predict_classes(frame)
        print (str(self.value_to_letter[prediction[0]]))

    def auc(self, y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        keras.backend.get_session().run(tf.local_variables_initializer())
        return auc
