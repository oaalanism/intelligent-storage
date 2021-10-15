import os
import sys
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

    def start(self):
        path = './output/'
        config = np.fromfile('config_storage.bin', dtype=np.int16)
        self.scope = [config[0], config[1]]
        self.minChange = config[2]
        if path:
            os.chdir(path)
            files = [f for f in os.listdir()]
            files = sorted(files, key=lambda x: float(x.split("-")[1][:-4]))
            for file in files:
                frameSparse = scipy.sparse.load_npz(file)
                self.frame =np.array(frameSparse.toarray())
                self.construct()
                self.show()
                ##self.show(frame)

    def __init__(self):
        self.lastFrame = np.array([])
        self.reconstruction = np.array([])
        self.scope = None
        self.minChange = None

reconstruction = VideoReconstruction()
reconstruction.start()