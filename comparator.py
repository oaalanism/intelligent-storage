import numpy as np
import sys
from videoReconstruction import VideoReconstruction
from decoderDepthColorized import DecoderDepthColorized
from rawDataReader import RawDataReader
import matplotlib.pyplot as plt


class Comparator: 

    def meanSquareDeviation(self, frameReference, frameConstruit):
        n = frameReference.size
        divi = np.ones((frameReference.shape))*1000
        diff = (frameReference - frameConstruit) / divi
        pot = np.power(diff, 2)
        sum = np.sum(pot, dtype=np.float64)
        msd = sum /n
        return msd
 
    def getDepthFrames(self):
        print("Getting Raw Data Frames...")
        self.raw_depth_frames = self.rawDataReader.start()
        print("Getting Algorithme Data Frames...")
        self.algo_depth_frames = self.algoReader.start()
        print("Getting Video Frames...")
        self.video_depth_frames = self.videoReader.start()

    def getMSDFrames(self, referenceFrames, contructionFrames):
        msdFrames = []
        t = 0
        if (len(referenceFrames) < len(contructionFrames)):
            t = len(referenceFrames)
        else:
            t = len(contructionFrames)
        for i in range(t):
            msd = self.meanSquareDeviation(referenceFrames[i], contructionFrames[i])
            msdFrames.append(msd)

        return msdFrames 
    
    def plotMSD(self):
        xRA = [i for i in range(len(self.msdRawAlgo))]
        xRV = [i for i in range(len(self.msdRawVideo))]
        plt.plot(xRA, self.msdRawAlgo, 'b')
        plt.plot(xRV, self.msdRawVideo, 'g')
        plt.legend(['MSD Raw-Algorithme', 'MSD Raw-Video'])
        plt.title("MSD")
        plt.show()

    def start(self):
        self.getDepthFrames()
        print("Calculating MSD between Raw and Algorithme Data...")
        self.msdRawAlgo = self.getMSDFrames(self.raw_depth_frames, self.algo_depth_frames)
        print("Calculating MSD between Raw and Video Data...")
        self.msdRawVideo = self.getMSDFrames(self.raw_depth_frames, self.video_depth_frames)
        print("Ploting MSD...")
        self.plotMSD()
        #x = str(input("Tap anything to scape"))

    def __init__(self, rawDataPath, algoPath, videoPath):
        self.rawDataReader = RawDataReader(rawDataPath)
        self.algoReader = VideoReconstruction(algoPath)
        self.videoReader = DecoderDepthColorized(videoPath)
