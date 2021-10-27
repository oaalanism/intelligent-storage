import os
from numpy.core.fromnumeric import sort
import scipy.sparse
import cv2
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from colorizeDepthImage import ColorizeDepthImage

class VideoReconstruction:

    def show(self):
        depth_colorized = self.colorize.appplyColorization(self.frame, 0, self.scope[1])
        cv2.imshow("Depth Stream", depth_colorized)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()

    def construct(self):
        
        if self.lastFrame.size == 0:
            self.lastFrame = self.frame
        else:
            one_sparse = self.frame.sign()
            self.lastFrame = self.lastFrame - self.lastFrame.multiply(one_sparse) + self.frame 

    def getFiles(self):
        self.files = [self.path + f for f in os.listdir(self.path)]
        self.files = sorted(self.files, key=lambda x: float(x.split("-")[1][:-4]))

    def getFileData(self):
        self.sparce_frames = []
        for file in self.files:
            print(file)
            frameSparse = scipy.sparse.load_npz(file)
            self.sparce_frames.append(frameSparse)

    def buildDepthFrames(self):
        self.depth_frames = []
        for sparce_frame in self.sparce_frames:
            self.frame = sparce_frame
            self.construct()
            self.depth_frames.append(self.lastFrame)
    
    def showVideo(self):
        for frame in self.depth_frames:
            self.frame = np.array(frame.toarray(), dtype="uint32")
            self.show()

    def start(self):
        self.getFiles()
        self.getFileData()
        self.buildDepthFrames()
        return self.depth_frames

    def __init__(self, path):
        self.lastFrame = np.array([])
        self.reconstruction = np.array([])
        self.scope = None
        self.minChange = None
        self.path = path
        config = np.fromfile('config_storage.bin', dtype='int32')
        self.scope = [config[0], config[1]]
        self.minChange = config[2]
        self.colorize = ColorizeDepthImage()