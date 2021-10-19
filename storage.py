import numpy as np
import scipy.sparse

class Storage: 
    """
    Test with Huffman
    def codeFrame(self):
        frameCoded = ""
        for i in range(self.frame.shape[0]):
            for j in range(self.frame.shape[1]):
                if self.frame[i, j] != 0:
                    code = self.huffmanCode.codeValue(self.frame[i, j])
                    frameCoded = frameCoded + code
        print (frameCoded)
        print (len(frameCoded))


    def calculateFrequency(self):
        unique, counts = np.unique(self.frame.flatten(), return_counts=True)
        self.frequency = np.asarray((unique, counts)).T
        self.frequency[::-1].sort(axis=0)
        self.huffmanCode = HuffmanCode(self.frequency)

    def startCodeFrame(self):
        self.calculateFrequency()
        self.codeFrame()
    """

    def storeDepthImage(self, nb_frame):
        depthImageSparce = scipy.sparse.csc_matrix(self.frame, dtype=np.int16)
        scipy.sparse.save_npz('./output/image-'+str(nb_frame)+'.npz', depthImageSparce)

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
        nb_pixels_changed = self.compare()

        if nb_pixels_changed > self.nb_pixels_max or time - self.inactivityTime < 10:
            #self.startCodeFrame()
            self.storeDepthImage(nb_frame)
            if nb_pixels_changed > self.nb_pixels_max:
                self.inactivityTime = time

        elif nb_pixels_changed < self.nb_pixels_max and self.inactivityTime == 9:
            self.inactivityTime = time

    def setFrame(self, frame):
        self.frame = frame

    def __init__(self, resolution, nb_pixels_max, scope, minChange):
        self.referenceData =  np.array([])
        self.width = resolution[0]
        self.height = resolution[1]
        self.nb_pixels_max = nb_pixels_max
        self.scope = scope
        self.minChange = minChange
        self.inactivityTime = 0
        self.huffmanCode = None