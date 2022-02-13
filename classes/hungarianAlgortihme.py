from typing import IO
import numpy as np
from numpy.core.fromnumeric import take
from sklearn.utils.linear_assignment_ import linear_assignment

from classes.detection import Detection
from classes.outils import Outils

class HungarianAlgorithme:
    """
    The Hungarian Algortihm object implemnt Hungarian algorithm for a set of trackers and detections to obtain the id association
    
    Object returns updated trackings
    
    Parameters
    ----------
    """
    def __init__(self):
        self.tracking = []
        self.outils = Outils()
        self.id = 1
    
    def hungarian(self, trackers, detections):
        """
        This function calculates IoU between each tracker and detection
        Parameters
        ----------
            trackers: Tracker Object
                Set of object trackers to calculate IoU
            detections: Array
                Set of array with the bounding boxes of detections  
        Returns
        -------
            matched_idx: Array
                Tracker's ID matched with detections
            unmatchedTrackers: Array
                Trackers not associated with a new detection
            unmatchedDetections: Array
                New Detections
        """
        nbDetections = len(detections)
        nbTrackers = len(trackers)

        unmatchedTrackers = []
        unmatchedDetections = []
        matched_idx = []

        IOUs = np.zeros((nbDetections, nbTrackers))

        for i in range(nbDetections):
            bbD = detections[i]
            for j in range(nbTrackers):
                bbT = trackers[j].KalmanFilter.X_
                #bbT = trackers[j]
                iou = self.outils.IOU(bbD, bbT)
                IOUs[i, j] = iou

        IOUm = np.zeros((nbDetections, nbTrackers))

        for d in range(nbDetections):
            if(np.count_nonzero(IOUs[d,:]) > 0):
                maxValue = np.amax(IOUs[d,:])
                for t in range(nbTrackers):
                    if(IOUs[d, t] == maxValue and np.amax(IOUs[:,t]) <= maxValue and np.count_nonzero(IOUm[:,t]) == 0):
                        IOUm[d,t] = 1
        """
        for j in range(nbTrackers):
            maxValue = np.amax(IOUs[:,j])

            if(maxValue != 0):
                index = np.where(IOUs[:,j] == maxValue)
                i = index[0]
                IOUm[i,j] = 1
        """
        if(np.count_nonzero(IOUm) > 0):
            matched_idx = linear_assignment(-IOUm)
        
        
            for t in range(nbTrackers):
                if(t not in matched_idx[:,1]):
                    unmatchedTrackers.append(trackers[t])

            for d in range(nbDetections):
                if(d not in matched_idx[:,0] ):
                    unmatchedDetections.append(detections[d])
            
        else:
            unmatchedTrackers = trackers
            unmatchedDetections = detections

        return matched_idx, unmatchedTrackers, unmatchedDetections

                        
                


    def findRelations(self, IOUs):
        candidates = np.zeros(IOUs.shape, dtype=bool)
        for i in range(IOUs.shape[0]):
            
            maxIOU = 0
            ind = None
            for j in range(IOUs.shape[1]):
                if maxIOU < IOUs[i, j]:
                   ind = i
                   maxIOU = IOUs[i, j]
            if(ind != None):
                candidates[i, ind] = True 

        return candidates
            
    def calculateIOUs(self, trackers, detections):
        lenTrackings = len(trackers)
        lenDetections = len(detections)
        IOUs = np.zeros((lenTrackings, lenDetections))

        for i in range(lenTrackings):
            for j in range(lenDetections):
                trackingBB = self.tracking[i].getBoundingBox()
                IOUs[i, j] = self.outils.IOU(trackingBB, detections[j])

        return IOUs

    def updateTrackings(self, detections,  candidates):
        toDelete = []
        for i in range(candidates.shape[0]):
            if(np.count_nonzero(candidates[i,:]) > 0):
                j = np.where(candidates[i,:])[0][0]
                self.tracking[i].updateBoundingBox(detections[j])
            else:
                self.tracking[i].timesNotDetected = self.tracking[i].timesNotDetected + 1 
                if(self.tracking[i].timesNotDetected > 3):
                    toDelete.append(i)
        
        for j in range(candidates.shape[1]):
            if(np.count_nonzero(candidates[:,j]) == 0):
                self.addTracking(detections[j])
                

    def addTracking(self, detection):
        d = Detection(self.id, detection)
        self.tracking.append(d)
        self.id = self.id + 1

    def addTrackings(self, detections):

        for detection in detections:
            self.addTracking(detection)
    
    def updateTrackers(self, detections):
        if(len(self.tracking) == 0):
            self.addTrackings(detections)
        elif(len(detections) > 0):
            IOUs = self.calculateIOUs(detections)
            candidates = self.findCandidates(IOUs)
            self.updateTrackings(detections, candidates)

        return self.tracking
