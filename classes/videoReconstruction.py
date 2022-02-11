import os
import scipy.sparse
import cv2
import numpy as np
import pyrealsense2 as rs
from matplotlib import cm as cm
from classes.colorizeDepthImage import ColorizeDepthImage
from classes.storage import Storage

class VideoReconstruction:
    """

    This object rebuild depth streaming from matrix sparce representation, also this object allows to save this frames in a video

    Parameters
    ----------
        path : str
            Directory path of version to rebuild

        saveVideo : false
            Variable to indicate if video will be saved
        
    """

    def __init__(self, path, save_video = False):
        self.lastFrame = np.array([])
        self.reconstruction = np.array([])
        self.scope = None
        self.minChange = None
        self.path = path + "algo/"
        config = np.fromfile(path+'config_storage.bin', dtype='int32')
        self.scope = [config[0], config[1]]
        self.minChange = config[2]
        self.nb_pixels_max = config[3]
        self.resolution = [config[4], config[4]] 
        self.colorize = ColorizeDepthImage()
        self.save_video = save_video
        self.start()
        

    def show(self):
        """
        Function to show video debuilded from sparse matrix

        Parameters
        ----------

        Returns
        -------
        """

        depth_colorized = self.colorize.appplyColorization(self.frame, 0, self.scope[1])
        cv2.imshow("Depth Stream", depth_colorized)
        key = cv2.waitKey(100)
        if key == 27:
            cv2.destroyAllWindows()

    def construct(self):
        """
        Function to build current frame using last frame

        Parameters
        ----------

        Returns
        -------
        """
        if self.lastFrame.size == 0:
            self.lastFrame = self.frame
        else:
            one_sparse = self.frame.sign()
            self.lastFrame = self.lastFrame - self.lastFrame.multiply(one_sparse) + self.frame 

    def getFiles(self):
        """
        Function to get name of sparce matrix files

        Parameters
        ----------

        Returns
        -------
        """
        self.files = [self.path + f for f in os.listdir(self.path)]
        self.files = sorted(self.files, key=lambda x: float(x.split("-")[1][:-4]))

    def getFileData(self):
        """
        Function to get sparce frames data

        Parameters
        ----------

        Returns
        -------
        """
        self.sparce_frames = []
        
        for file in self.files:
            frameSparse = scipy.sparse.load_npz(file)
            self.sparce_frames.append(frameSparse)

    def buildDepthFrames(self):
        """
        Function to build streaming from sparce matrix files

        Parameters
        ----------

        Returns
        -------
        """
        self.depth_frames = []
        for sparce_frame in self.sparce_frames:
            self.frame = sparce_frame
            self.construct()
            self.depth_frames.append(self.lastFrame)

    def saveVideo(self):
        """
        Function to save video

        Parameters
        ----------

        Returns
        -------
        """
        self.storage = Storage(self.resolution, self.nb_pixels_max, self.scope, self.minChange)
        print("saving algorithm video")
        for frame in self.depth_frames:
            self.frame = np.array(frame.toarray(), dtype="uint32")
            depth_colorized = self.colorize.appplyColorization(self.frame, 0, self.scope[1])

            self.storage.depth_color_image = depth_colorized
            self.storage.storeVideo()

        print("end...")


    def showVideo(self):

        """
        Function to display streaming from sparce matrix files

        Parameters
        ----------

        Returns
        -------
        """
        
        for frame in self.depth_frames:
            self.frame = np.array(frame.toarray(), dtype="uint32")
            depth_colorized = self.colorize.appplyColorization(self.frame, 0, self.scope[1])
            cv2.imshow("Depth Stream", depth_colorized)
            
            key  = cv2.waitKey(1)

            if key == ord('q'):
                return False
            if key == ord('p'):
                cv2.waitKey(-1)


    def start(self):
        """
        Function start building algorithme

        Parameters
        ----------

        Returns
        -------
        """
        self.getFiles()
        self.getFileData()
        self.buildDepthFrames()
        if(self.save_video):
            self.saveVideo()
        return self.depth_frames

    