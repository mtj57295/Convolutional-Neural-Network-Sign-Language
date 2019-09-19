import cv2
import numpy as np
from cnn import CNN
import os

class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.nn = CNN()
        self.model = None
        if os.path.exists('./model.h5'):
            print 'Model Already Exist'
            print '--------------------'
            self.model = self.nn.load_model()
        else:
            print 'Model Does Not Exist'
            print '--------------------'
            self.model = self.nn.create_cnn_model()

    def __del__(self):
        self.video.release()

    def get_frame(self):

        _, fr = self.video.read()
        frame = np.copy(fr)
        frame = self.nn.data_frame(frame)
        self.nn.predict_frame(self.model, frame)
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
