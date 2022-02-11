import numpy as np
import scipy.sparse
import cv2
from classes.colorizeDepthImage import ColorizeDepthImage
import os
import pathlib

class Storage:

    """
    Storage object store depth data frame in the three representations 
    ------ Raw Data ------------------------
    ------ Colorized Depth frame -----------
    ------ Matrix sparse -------------------
    Representations are stored in output repository that is automaticaly created if it does not exist in the current directory
    
    
    Then, object will create another repository with the designation:
    --------------- vXX -----------------------------------

    Where XX is a number that object increments automatically depending on the number of the last repository. 
    If any vesions has been stored object creat new one with number 1

    Parameters of streaming are stored in config_storage.bin to build strealing using the sparce matrix representation

    Parameters
    ----------
    resolution : Array
        Dimentions of depth frames
        One dimention array with two variables, first is the width and second is the lenght of an image

    nb_pixels_max : Int
        Set the minimum number of pixels changed to consider that an image is different from other

    scope : Array
        Scope is a parameter necessary to set the scope depth data to store
        One dimention array with two variables, first argument set min distance and second max distance

    min_change : Int
        Set smallest difference between two pixels to consider that they have changed, distance in mili meters
    """
    def __init__(self, resolution, nb_pixels_max, scope, min_change):
        self.preparWorkSpace()
        self.referenceData =  np.array([])
        self.width = resolution[0]
        self.height = resolution[1]
        self.nb_pixels_max = nb_pixels_max
        self.scope = scope
        self.min_change = min_change
        self.inactivityTime = 9
        self.colorizeDepth = ColorizeDepthImage()
        self._name = self.videoDir+"stream.avi"
        self._fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(self._name, self._fourcc, 20.0, (self.width, self.height))
        self.depth_color_image = None
        c = [self.scope[0], self.scope[1], self.min_change, self.nb_pixels_max, self.width, self.height]                        
        config = np.array(c)
        config.astype('int32').tofile(self.outputDir+'config_storage.bin')

    def storeDepthImage(self, nb_frame):
        """
        Function strore depth image in sparse

        Parameters
        ----------
        nb_frame : Int
            Frame number in the streaming

        Returns
        -------

        """
        depth_image_sparse = scipy.sparse.csc_matrix(self.frame, dtype='uint16')
        scipy.sparse.save_npz(self.algoDir+"image-"+str(nb_frame)+".npz", depth_image_sparse)

    def compare(self):
        """
        Function to compare precendent frame with current

        Parameters
        ----------

        Returns
        -------
        nb_pixels_changed : Int
            Number of pixels that have changed between the current and the last frame.

        """
        nb_pixels_changed = 0
        if self.referenceData.size == 0:                                                                                    # | Get the first frame of version
            nb_pixels_changed = self.width*self.height                                                                      # |
            self.referenceData = self.frame                                                                                 # |
        else:                                                                                                               # |
            diff = np.abs(np.subtract(self.referenceData, self.frame))                                                      # | Get difference between current and last frame
            minTolerance = np.ones((self.height, self.width))*self.min_change                                               # |
            com = diff >= minTolerance                                                                                      # | Compare current frame to get those pixels higher 
                                                                                                                            # | or equal to the minimum tolerance distance
            (_, counts) = np.unique(com, return_counts=True)                                                                # | Count pixels that has changed 

            nb_pixels_changed = counts[1]
            zero = np.zeros((self.height, self.width))
            self.referenceData = self.frame
            self.frame = np.where(com, self.frame, zero)
            
        
        return nb_pixels_changed                                                    

    def applyThreshold(self):
        """
        Apply threshold for the current frame

        Parameters
        ----------

        Returns
        -------
        """
        self.frame = np.where(np.logical_and(self.scope[0] < self.frame, self.frame < self.scope[1] ), self.frame, 0)
                      
    def store(self, time, nb_frame):
        """
        Function to save frame data if downtime is not exceeded

        Parameters
        ----------
        time : Int
            Current time of video
        nb_frame : Int
            Frame number in streaming

        Returns
        -------
        """
        self.applyThreshold()
        self.depth_color_image = self.colorizeDepth.appplyColorization(self.frame, self.scope[0], self.scope[1])
        self.frameButre = self.frame
        nb_pixels_changed = self.compare()
        if nb_pixels_changed > self.nb_pixels_max or time - self.inactivityTime < 10:                                           # | Store data if number of pixels 
            self.storeDepthImage(nb_frame)                                                                                      # | changed between both frames are higher 
            self.storeRawData(nb_frame)                                                                                         # | than the minimum tolerance or 
            self.storeVideo()                                                                                                   # | downtime is less than 10s
            if nb_pixels_changed > self.nb_pixels_max:
                self.inactivityTime = time

        elif nb_pixels_changed < self.nb_pixels_max and self.inactivityTime == 9:                                               # | Data storage is paused
            self.inactivityTime = time                                                                                          # | if downtime passed 10s

    def storeRawData(self, nb_frame):
        """
        Function to store raw data in numpy binary format

        Parameters
        ----------
        nb_frame : Int 
            Frame number in streaming

        """

        with open(self.rawDir+"image-"+str(nb_frame)+".npy", "wb") as f:
            np.save(f, self.frameButre)
        
    def storeVideo(self):
        """
        Store colorized representation in a video

        Parameters
        ----------

        Returns
        -------
        """
        self._out.write(self.depth_color_image)

    def stopRecordinfVideo(self):
        """
        Stop video store 

        Parameters
        ----------

        Returns
        -------
        """

        self._out.release()

    def setColorizeDepth(self, depth_color_image):
        """
        Function to set colorized depth image

        Parameters
        ----------

        Returns
        -------
        """
        self.depth_color_image = depth_color_image

    def setFrame(self, frame):
        """
        Function to set current depth frame 

        Parameters
        ----------

        Returns
        -------
        """
        self.frame = frame

    def preparWorkSpace(self):

        """
        Function to create directories where data is stored

        Parameters
        ----------

        Returns
        -------
        """

        file = pathlib.Path("./output/")
        if not(file.exists ()):
            os.mkdir("./output/")

        dirs = []
        for dir in os.listdir("./output/"):
            if (dir.find("v") != -1):
                dirs.append(dir)
        last = 1
        if len(dirs) > 0:
            dirs = sorted(dirs, key=lambda x: float(x.split("v")[1]))
            last = int(dirs[len(dirs)-1].split("v")[1])+1

        self.outputDir = "./output/v"+str(last)+"/"
        self.algoDir = self.outputDir+"algo/"
        self.rawDir = self.outputDir+"raw_data/"
        self.videoDir = self.outputDir+"video/"
        self.positionsDir = self.outputDir+"positions/"

        
        os.mkdir(self.outputDir)
        os.mkdir(self.algoDir)
        os.mkdir(self.rawDir)
        os.mkdir(self.videoDir)
        os.mkdir(self.positionsDir)
            

    