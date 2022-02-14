import os
import numpy as np

class RawDataReader:
    """
        This object reads raw data format
        
        Parameters
        ----------
    """
    def __init__(self, path):
        self.path = path
        
    def getFiles(self):
        """
            Function to get the names of raw data files 
            
            Parameters
            ----------
            
            Returns
            -------
        """
        self.files = [self.path + f for f in os.listdir(self.path)]
        self.files = sorted(self.files, key=lambda x: float(x.split("-")[1][:-4]))


    def getFilesNumpyData(self):
        """
            Extract depth data from files
            
            Parameters
            ----------
            
            Returns
            -------
        """
        self.depth_frames = []

        for file in self.files:
            frame = np.load(file)
            self.depth_frames.append(frame)

        

    def start(self):
        """
            function that starts data extraction 
            
            Parameters
            ----------
            
            Extract
            -------
        """
        self.getFiles()
        self.getFilesNumpyData()
        return self.depth_frames

