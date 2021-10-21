import numpy as np
import sys
from videoReconstruction import VideoReconstruction
from decoderDepthColorized import DecoderDepthColorized
from rawDataReader import RawDataReader
import matplotlib.pyplot as plt


class Comparator: 

    def meanSquareDeviation(self, frameReference, frameConstruit):
        n = frameReference.size
        diff = pow(abs(frameReference - frameConstruit), 2)
        sum = diff.sum()
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
        for i in range(len(referenceFrames)):
            msd = self.meanSquareDeviation(referenceFrames[i], contructionFrames[i])
            msdFrames.append(msd)

        return msdFrames 
    
    def plotMSD(self):
        x = [i for i in range(len(self.msdRawAlgo))]
        plt.plot(x, self.msdRawAlgo, label ='MSD Raw-Algorithme')
        plt.plot(x, self.msdRawVideo, label='MSD Raw-Video')
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
        x = str(input("Tap anything to scape"))

    def __init__(self, rawDataPath, algoPath, videoPath):
        self.rawDataReader = RawDataReader(rawDataPath)
        self.algoReader = VideoReconstruction(algoPath)
        self.videoReader = DecoderDepthColorized(videoPath)
