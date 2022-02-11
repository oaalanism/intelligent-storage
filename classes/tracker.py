from classes.kalmanFilter import KalmanFilter
import numpy as np


class Tracker:

    def delete(self, nbLastImage):
        self.nbLastImage = nbLastImage

    def associate(self, newCentroids):
        if(len(newCentroids) > 0):
            x1 = self.centroid[0]
            y1 = self.centroid[1]

            distances = np.sqrt(np.abs(newCentroids[:,0] - x1)**2 + np.abs(newCentroids[:,1] - y1)**2)

            ind = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

            if(distances[ind] <= 100):
                newCentroid = newCentroids[ind]
                self.centroid = newCentroid
                newCentroids = np.delete(newCentroids, ind, 0)
                return True, newCentroids
        
        return False, newCentroids


    def update(self, Z):
        self.KalmanFilter.update(Z)
        self.boundingBox[0] = Z[0]
        self.boundingBox[1] = Z[1]
        self.boundingBox[2] = Z[2]
        self.boundingBox[3] = Z[3]

    def predict(self):
        self.detected = False
        self.KalmanFilter.predict()

    def __init__(self, id = None, boundingBox = None, nbFirstImage = 0, centroid = None):
        self.id = id
        self.boundingBox = boundingBox
        self.detected = False
        self.timesUndectected = 0
        self.nbFirstImage = nbFirstImage
        self.nbLastImage = None
        if(boundingBox != None):
            self.KalmanFilter = KalmanFilter(boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3])
        else:
            self.KalmanFilter = None
        self.centroid = centroid