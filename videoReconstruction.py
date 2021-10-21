import os
from numpy.core.fromnumeric import sort
import scipy.sparse
import cv2
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from matplotlib import cm as cm

class VideoReconstruction:

    def show(self):
        qm = plt.pcolormesh(self.frame, cmap='nipy_spectral', vmin=0, )
        rgba = qm.to_rgba(qm.get_array().reshape(self.frame.shape))
        cv2.imshow("Depth Stream", rgba)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
        #plt.show()

    def construct(self):
        
        if self.lastFrame.size == 0:
            self.lastFrame = self.frame
            self.reconstruction = self.frame
        else:
            reconstruction = np.where(self.frame == 0, self.lastFrame, self.frame)
            self.lastFrame = reconstruction
            self.frame = reconstruction

    def getFiles(self):
        self.files = [self.path + f for f in os.listdir(self.path)]
        self.files = sorted(self.files, key=lambda x: float(x.split("-")[1][:-4]))

    def getFileData(self):
        self.sparce_frames = []
        for file in self.files:
            frameSparse = scipy.sparse.load_npz(file)
            frame =np.array(frameSparse.toarray())
            self.sparce_frames.append(frame)

    def buildDepthFrames(self):
        self.depth_frames = []
        for sparce_frame in self.sparce_frames:
            self.frame = sparce_frame
            self.construct()
            self.depth_frames.append(self.lastFrame)
    
    def showVideo(self):
        for frame in self.depth_frames:
            self.frame = frame
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
        config = np.fromfile('config_storage.bin', dtype=np.int16)
        self.scope = [config[0], config[1]]
        self.minChange = config[2]

#reconstruction = VideoReconstruction('./output/algo/')
#reconstruction.getFiles()
#reconstruction.getFileData()
#reconstruction.showVideo()
#reconstruction.start()