import numpy as np
import cv2

class DecoderDepthColorized:

    def buildImage(self, frame):
        """
            frame: (bgr) numpy 3D
        """
        prb = frame[:,:,0]
        prg = frame[:,:,1]
        prr = frame[:,:,2]
        filtre1 = np.where(np.bitwise_and(np.bitwise_and(prr >= prg, prr >= prb), prg >= prb), prg - prb, 0)
        filtre2 = np.where(np.bitwise_and(np.bitwise_and(prr >= prg, prr >= prb), prg < prb), prg - prb + 1529, 0)
        filtre3 = np.where(np.bitwise_and(prg >= prr, prg >= prb), prb - prr + 510, 0)
        filtre4 = np.where(np.bitwise_and(prb >= prg, prb >= prr), prr-prg +1020, 0)
        d_rnormal = filtre1 + filtre2 + filtre3 + filtre4
        d_recovery = self.d_min + ((self.d_max - self.d_min)*d_rnormal/1529)
        return d_recovery


    def transformVideoToNumpy(self):
        cap = cv2.VideoCapture(self.video)

        success = True
        self.frames = []

        while success:
            success, image = cap.read()
            if success:
                self.frames.append(image)

    def transformFramesToDepth(self):
        self.depth_frames = []
        for frame in self.frames:
            depth_frame = self.buildImage(frame)
            self.depth_frames.append(depth_frame)

    def start(self):
        self.transformVideoToNumpy()
        self.transformFramesToDepth()

        return self.depth_frames

    def __init__(self, video):
        config = np.fromfile('config_storage.bin', dtype=np.int16)
        self.d_min = config[0]
        self.d_max = config[1]
        self.video = video