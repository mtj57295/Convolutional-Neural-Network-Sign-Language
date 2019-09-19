from cnn import CNN
import os
import time
import threading
import tensorflow as tf

nn = CNN()

if os.path.exists('./model2.h5'):
    print 'Model Already Exist'
    print '--------------------'
    model = nn.load_model()
else:
    print 'Model Does Not Exist'
    print '--------------------'
    model = nn.create_cnn_model()

input = int(raw_input('Enter 0 to train:\nEnter 1 to predict:'))

if input == 0:
    list = []
    for i in xrange(65, 91):
        print 'Training:    ' + str(chr(i))
        print '-----------------------------------------------------'
        images = nn.fetch_data('../asl_alphabet_train/' + chr(i))
        nn.prepare_data(images, letter=chr(i))
    #Nothing
    images = nn.fetch_data('../asl_alphabet_train/' + 'nothing')
    nn.prepare_data(images, letter='nothing')
    #Space
    images = nn.fetch_data('../asl_alphabet_train/' + 'space')
    nn.prepare_data(images, letter='space')

    inputs, labels = nn.mix_data()
    nn.train_cnn_model(model, inputs, labels)
elif input == 1:
    images = nn.fetch_data('../asl_alphabet_test')
    inputs, labels = nn.prepare_test_data(images)
    nn.predict_sample(model, inputs, labels)

nn.save_model(model)
