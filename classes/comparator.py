import numpy as np
from classes.videoReconstruction import VideoReconstruction
from classes.decoderDepthColorized import DecoderDepthColorized
from classes.rawDataReader import RawDataReader
import matplotlib.pyplot as plt
import os


class Comparator: 

    """
    Object Extractor extract detection from a set of depth data to save each one in a csv file with path ./data/detection/vXX/vXX.csv
    And a video of depth grey frame with detections

    
    The data saved is :
        Number of version
        Number of image
        Person id
        x coordinate center of the bounding box detection
        y coordinate center of the bounding box detection 
        width of bounding box
        heigt of bounding box
        maximum distance of detection
    

    Parameters
    ----------
        version : Int
            Number of data version to detect people
        nb_ref_frame : Int
            Reference number of frame to extract background
    """

    def meanSquareDeviation(self, frameReference, frameConstruit):
        n = frameReference.shape[0] * frameReference.shape[1]
        diff = (frameReference - frameConstruit)*0.001
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
        plt.xlabel("frame")
        plt.ylabel("MSE")
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

    def getLastOuputDir(self):
        dirs = []
        for dir in os.listdir("."):
            if (dir.find("output.v") != -1):
                dirs.append(dir)
        last = 1
        if len(dirs) > 0:
            dirs = sorted(dirs, key=lambda x: float(x.split("v")[1]))
            last = int(dirs[len(dirs)-1].split("v")[1])

        return "./output.v"+str(last)+"/"

    def __init__(self, outputPath = None):

        if outputPath == None :
            print("Getting last output directory")
            outputPath = self.getLastOuputDir()
        
        rawDataPath = outputPath + "raw_data/"
        algoPath = outputPath + "algo/"
        videoPath = outputPath + "video/"
        self.rawDataReader = RawDataReader(rawDataPath)
        self.algoReader = VideoReconstruction(algoPath)
        self.videoReader = DecoderDepthColorized(videoPath)
