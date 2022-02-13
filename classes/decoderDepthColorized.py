import numpy as np
import cv2

class DecoderDepthColorized:
    """
    Decoder object transform matrix sparce representation in depth frames using recovery from colorized image
    
    --------------------------------https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras----------------------

    Parameters
    ----------
        version_path : Str
            Repository to get data 
    """
    
    def __init__(self, version_path):
        video = version_path + "/video/stream.avi"
        config = np.fromfile(version_path+'config_storage.bin', dtype='int32')
        self.d_min = config[0]
        self.d_max = config[1]
        self.video = video

    def buildImage(self, frame):
        """
        Function to build depth image from RGB videos
        Parameters
        ----------
            frame : Array 
                RGB frame
        Returns
        -------
            d-recovery : Array
                depth frame recovery
        
        """
        #frame = frame.astype('int8')
        
        prb = frame[:,:,0]
        prg = frame[:,:,1]
        prr = frame[:,:,2]
        filtre1 = np.where(np.bitwise_and(np.bitwise_and(prr >= prg, prr >= prb), prg >= prb), prg - prb, 0)
        filtre2 = np.where(np.bitwise_and(np.bitwise_and(prr >= prg, prr >= prb), prg < prb), prg - prb + 1529, 0)
        filtre3 = np.where(np.bitwise_and(prg >= prr, prg >= prb), prb - prr + 510, 0)
        filtre4 = np.where(np.bitwise_and(prb >= prg, prb >= prr), prr-prg +1020, 0)
        d_rnormal = filtre1 + filtre2 + filtre3 + filtre4
        d_recovery = self.d_min + (((self.d_max - self.d_min)/1529)*d_rnormal)
        return d_recovery

    def transformFramesToDepth(self):
        """
        function to loop into frames video and transform it in depth frames
         Parameters
        ----------
        Returns
        -------
        """
        self.depth_frames = []
        cap = cv2.VideoCapture(self.video)
        success, image = cap.read()
        while success:
            depth_frame = self.buildImage(image)
            self.depth_frames.append(depth_frame)

    def start(self):
        """
        Principal function to transform rgb representation
        Parameters
        ----------
        Returns
        -------
        """
        
        self.transformFramesToDepth()

        return self.depth_frames

    
