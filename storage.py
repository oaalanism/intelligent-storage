import numpy as np
import scipy.sparse
import cv2
from colorizeDepthImage import ColorizeDepthImage
import os

class Storage: 

    def storeDepthImage(self, nb_frame):
        depthImageSparce = scipy.sparse.csc_matrix(self.frame, dtype='uint16')
        scipy.sparse.save_npz(self.algoDir+"image-"+str(nb_frame)+".npz", depthImageSparce)

    def compare(self):
        nb_pixels_changed = 0
        if self.referenceData.size == 0:
            nb_pixels_changed = self.width*self.height
            self.referenceData = self.frame
        else:
            diff = np.abs(np.subtract(self.referenceData, self.frame))
            minTolerance = np.ones((self.height, self.width))*self.minChange 
            com = diff >= minTolerance
            com_neg = diff < minTolerance
            (_, counts) = np.unique(com, return_counts=True)

            nb_pixels_changed = counts[1]
            zero = np.zeros((self.height, self.width))
            self.referenceData = self.frame
            self.frame = np.where(com, self.frame, zero)
            
        
        return nb_pixels_changed

    def applyThreshold(self):
        self.frame = np.where(np.logical_and(self.scope[0] < self.frame, self.frame < self.scope[1] ), self.frame, 0)
                      
    def store(self, time, nb_frame):
        self.applyThreshold()
        self.depth_color_image = self.colorizeDepth.appplyColorization(self.frame, self.scope[0], self.scope[1])
        self.frameButre = self.frame
        nb_pixels_changed = self.compare()
        if nb_pixels_changed > self.nb_pixels_max or time - self.inactivityTime < 10:
            self.storeDepthImage(nb_frame)
            self.storeRawData(nb_frame)
            self.storeVideo()
            if nb_pixels_changed > self.nb_pixels_max:
                self.inactivityTime = time

        elif nb_pixels_changed < self.nb_pixels_max and self.inactivityTime == 9:
            self.inactivityTime = time

    def storeRawData(self, nb_frame):
        with open(self.rawDir+"image-"+str(nb_frame)+".npy", "wb") as f:
            np.save(f, self.frameButre)
        
    def storeVideo(self):
        self._out.write(self.depth_color_image)

    def stopRecordinfVideo(self):
        self._out.release()

    def setColorizeDepth(self, depth_color_image):
        self.depth_color_image = depth_color_image

    def setFrame(self, frame):
        self.frame = frame

    def preparWorkSpace(self):

        dirs = []
        for dir in os.listdir("."):
            if (dir.find("output.v") != -1):
                dirs.append(dir)
        last = 1
        if len(dirs) > 0:
            dirs = sorted(dirs, key=lambda x: float(x.split("v")[1]))
            last = int(dirs[len(dirs)-1].split("v")[1])+1

        self.outputDir = "./output.v"+str(last)+"/"
        self.algoDir = self.outputDir+"algo/"
        self.rawDir = self.outputDir+"raw_data/"
        self.videoDir = self.outputDir+"video/"
        os.mkdir(self.outputDir)
        os.mkdir(self.algoDir)
        os.mkdir(self.rawDir)
        os.mkdir(self.videoDir)
            

    def __init__(self, resolution, nb_pixels_max, scope, minChange):
        self.preparWorkSpace()
        self.referenceData =  np.array([])
        self.width = resolution[0]
        self.height = resolution[1]
        self.nb_pixels_max = nb_pixels_max
        self.scope = scope
        self.minChange = minChange
        self.inactivityTime = 9
        self.colorizeDepth = ColorizeDepthImage()
        self._name = self.videoDir+"stream.avi"
        self._fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(self._name, self._fourcc, 20.0, (424, 240))
        self.depth_color_image = None
        c = [self.scope[0], self.scope[1], self.minChange]
        config = np.array(c)
        config.astype('int32').tofile('config_storage.bin')