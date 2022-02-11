import numpy as np
from classes.videoReconstruction import VideoReconstruction
from classes.decoderDepthColorized import DecoderDepthColorized
from classes.rawDataReader import RawDataReader
import matplotlib.pyplot as plt
import os


class Comparator: 

    """
    Object Comparator compare precition between three representations
    ------ Raw Data ------------------------
    ------ Colorized Depth frame -----------
    ------ Matrix sparse -------------------

    This Object use MSD as parameter. 

    Where data reference is Raw Data representation
    At the end this object shows a graph with this parameter for each frame.

    Parameters
    ----------
        output_path : Str
            Repository to get data 
    """

    def __init__(self, output_path = None):

        if output_path == None :
            print("Getting last output directory")
            output_path = self.getLastOuputDir()
        
        rawDataPath = output_path + "raw_data/"
        algoPath = output_path + "algo/"
        videoPath = output_path + "video/"
        self.rawDataReader = RawDataReader(rawDataPath)
        self.algoReader = VideoReconstruction(algoPath)
        self.videoReader = DecoderDepthColorized(videoPath)

    def meanSquareDeviation(self, frameReference, frameConstruit):

        """
        Function to calculate MSD between two frames

        Parameters
        ----------
            frameReference : Array
                Reference frame

            frameConstruit : Array
                frame to evaluate

        Returns
        -------

        """
        n = frameReference.shape[0] * frameReference.shape[1]
        diff = (frameReference - frameConstruit)*0.001
        pot = np.power(diff, 2)
        sum = np.sum(pot, dtype=np.float64)
        msd = sum /n
        return msd
 
    def getDepthFrames(self):
        """
        Function to get depth frames from three representations

        Parameters
        ----------

        Returns
        -------

        """
        print("Getting Raw Data Frames...")
        self.raw_depth_frames = self.rawDataReader.start()
        print("Getting Algorithme Data Frames...")
        self.algo_depth_frames = self.algoReader.start()
        print("Getting Video Frames...")
        self.video_depth_frames = self.videoReader.start()

    def getMSDFrames(self, referenceFrames, contructionFrames):
        """
        Function to get MSD for each data 

        Parameters
        ----------

        Returns
        -------

        """
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
        """
        Function to plot MSD

        Parameters
        ----------

        Returns
        -------

        """
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
        """
        Function to start to get MSD

        Parameters
        ----------

        Returns
        -------

        """
        self.getDepthFrames()
        print("Calculating MSD between Raw and Algorithme Data...")
        self.msdRawAlgo = self.getMSDFrames(self.raw_depth_frames, self.algo_depth_frames)
        print("Calculating MSD between Raw and Video Data...")
        self.msdRawVideo = self.getMSDFrames(self.raw_depth_frames, self.video_depth_frames)
        print("Ploting MSD...")
        self.plotMSD()
        #x = str(input("Tap anything to scape"))

    def getLastOuputDir(self):
        """
        Function to get last output directory version

        Parameters
        ----------

        Returns
        -------

        """
        dirs = []
        for dir in os.listdir("."):
            if (dir.find("./output/v") != -1):
                dirs.append(dir)
        last = 1
        if len(dirs) > 0:
            dirs = sorted(dirs, key=lambda x: float(x.split("v")[1]))
            last = int(dirs[len(dirs)-1].split("v")[1])

        return "./output/v"+str(last)+"/"

   
