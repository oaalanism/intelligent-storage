import numpy as np
import scipy.sparse
import cv2
from colorizeDepthImage import ColorizeDepthImage

class Storage: 

    def storeDepthImage(self, nb_frame):
        depthImageSparce = scipy.sparse.csc_matrix(self.frame, dtype=np.int16)
        scipy.sparse.save_npz('./output/algo/image-'+str(nb_frame)+'.npz', depthImageSparce)

    def compare(self):

        if self.referenceData.size == 0:
            c = [self.scope[0], self.scope[1], self.minChange]
            config = np.array(c)
            config.astype('int16').tofile('config_storage.bin')
            nb_pixels_changed = self.width*self.height
            self.referenceData = self.frame
        else:
            diff = abs(self.referenceData - self.frame)
            one = np.ones((self.height, self.width))*self.minChange 
            com = diff > one
            com_neg = diff < one
            (_, counts) = np.unique(com, return_counts=True)

            nb_pixels_changed = counts[1]
            
            self.referenceData = self.frame
            self.frame = np.where(com_neg, 0, self.frame)
            
        
        return nb_pixels_changed

    def applyThreshold(self):
        self.frame = np.where(np.logical_and(self.scope[0] < self.frame, self.frame < self.scope[1] ), self.frame, 0)
                      
    def store(self, time, nb_frame):
        self.applyThreshold()
        self.depth_color_image = self.colorizeDepth.appplyColorization(self.frame, self.scope[0], self.scope[1])
        nb_pixels_changed = self.compare()

        if nb_pixels_changed > self.nb_pixels_max or time - self.inactivityTime < 10:
            #self.startCodeFrame()
            self.storeDepthImage(nb_frame)
            self.storeRawData(nb_frame)
            self.storeVideo()
            if nb_pixels_changed > self.nb_pixels_max:
                self.inactivityTime = time

        elif nb_pixels_changed < self.nb_pixels_max and self.inactivityTime == 9:
            self.inactivityTime = time

    def storeRawData(self, nb_frame):
        with open('./output/raw_data/image-'+str(nb_frame)+".npy", "wb") as f:
            np.save(f, self.frame)

    def buildRedPixel(self, d_normal):
        filtre1 = np.where(np.bitwise_or(np.bitwise_and(0 <= d_normal, d_normal <= 255), np.bitwise_and(1275 <= d_normal, d_normal <= 1529)), 255, 0)
        filtre2 = np.where(np.bitwise_and(255 < d_normal, d_normal <= 510), 255 - d_normal, 0)
        filtre3 = np.where(np.bitwise_and(510 < d_normal, d_normal <= 1020), 0, 0)
        filtre4 = np.where(np.bitwise_and(1020 < d_normal, d_normal <= 1275), d_normal - 1020, 0)
        pr = filtre1 + filtre2 + filtre3 + filtre4
        return pr

    def buildGreenPixel(self, d_normal):
        filtre1 = np.where(np.bitwise_and(0 < d_normal, d_normal <= 255), d_normal, 0)
        filtre2 = np.where(np.bitwise_and(255 < d_normal, d_normal <= 510), 255, 0)
        filtre3 = np.where(np.bitwise_and(510 <d_normal, d_normal <= 765), 765 - d_normal, 0)
        filtre4 = np.where(np.bitwise_and(765 < d_normal, d_normal <= 1529), 0, 0)
        pg = filtre1 + filtre2 + filtre3 + filtre4
        return pg

    def buildBluePixel(self, d_normal):
        filtre1 = np.where(np.bitwise_and(0 < d_normal, d_normal <= 765), 0, 0)
        filtre2 = np.where(np.bitwise_and(765 < d_normal, d_normal <= 1020), d_normal - 765, 0)
        filtre3 = np.where(np.bitwise_and(1020 < d_normal, d_normal <= 1275), 255, 0)
        filtre4 = np.where(np.bitwise_and(1275 < d_normal, d_normal <= 1529), 1275 - d_normal, 0)
        pb = filtre1 + filtre2 + filtre3 + filtre4
        return pb

    def buildImage(self, d_normal):
        pr = self.buildRedPixel(d_normal)
        pg = self.buildGreenPixel(d_normal)
        pb = self.buildBluePixel(d_normal)

        image = np.zeros((self.height, self.width, 3))
        image[:,:,2] = pr
        image[:,:,1] = pg
        image[:,:,0] = pb
        image = image.astype('uint8')
        return image

    def appplyColorization(self):

        d_normal = ((self.frame - self.scope[0]) / (self.scope[1] - self.scope[0])) * 1529

        self.depth_color_image = self.buildImage(d_normal)
        return self.depth_color_image
        
        
        
        
    def storeVideo(self):
        self._out.write(self.depth_color_image)

    def stopRecordinfVideo(self):
        self._out.release()

    def setColorizeDepth(self, depth_color_image):
        self.depth_color_image = depth_color_image

    def setFrame(self, frame):
        self.frame = frame

    def __init__(self, resolution, nb_pixels_max, scope, minChange):
        self.referenceData =  np.array([])
        self.width = resolution[0]
        self.height = resolution[1]
        self.nb_pixels_max = nb_pixels_max
        self.scope = scope
        self.minChange = minChange
        self.inactivityTime = 9
        self.colorizeDepth = ColorizeDepthImage()
        self._name = "./output/video/stream.avi"
        self._fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(self._name, self._fourcc, 20.0, (424, 240))
        self.depth_color_image = None