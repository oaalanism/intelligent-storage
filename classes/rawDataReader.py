import os
import numpy as np

class RawDataReader:

    def getFiles(self):
        self.files = [self.path + f for f in os.listdir(self.path)]
        self.files = sorted(self.files, key=lambda x: float(x.split("-")[1][:-4]))


    def getFilesNumpyData(self):
        self.depth_frames = []

        for file in self.files:
            frame = np.load(file)
            self.depth_frames.append(frame)

        

    def start(self):
        self.getFiles()
        self.getFilesNumpyData()
        return self.depth_frames

    def __init__(self, path):
        self.path = path