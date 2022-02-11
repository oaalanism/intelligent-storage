import numpy as np

from classes.tracker import Tracker
from classes.hungarianAlgortihme import HungarianAlgorithme

class MultiTracker:
    def updateTrackerNotDetected(self, tracker):
        tracker.timesUndectected = tracker.timesUndectected + 1
        if tracker.timesUndectected > 15:
            tracker.delete(self.nbImage)
            self.nonDetected.append(tracker)
            self.trackers.remove(tracker)

    def updateTrackersNotDetected(self, trackersNonDetected):
        for tracker in trackersNonDetected:
            self.updateTrackerNotDetected(tracker)

    def associateTrackers(self, detections):
        detections = np.array(detections)
        trackersNotAssociated = []
        detectionsNotAssociated = detections
        trackersAssociated = []
        self.nonDetected = []
        for tracker in self.trackers:
            associated, detectionsNotAssociated = tracker.associate(detections)
            if(associated):
                trackersAssociated.append(tracker)
            else:
                trackersNotAssociated.append(tracker)


        for centroid in detectionsNotAssociated:
            tracker = Tracker()
            tracker.centroid = centroid
            tracker.id = self.id 
            self.trackers.append(tracker)
            self.id = self.id + 1

        self.updateTrackersNotDetected(trackersNotAssociated)

        return self.trackers, trackersNotAssociated, detectionsNotAssociated 
        """
        associated = False
        for tracker in self.trackers:
            detectionsBool = np.zeros(len(newDetections), dtype=bool)
            associated, indice = tracker.associate(newDetections)
            if(not(associated)):
                #newDetections.append(detection)
                self.updateTrackerNotDetected(tracker)
            else:
                newDetectionsBool[indice] = True

            newDetections = newDetections[np.where(np.bitwise_not(newDetectionsBool))]
        self.addNewTrackers(newDetections)
        """
        
    def getPredictions(self):
        predictions = []
        for tracker in self.trackers:
            predictions.append(tracker.KalmanFilter.X_)
        return predictions

    def predictTrackers(self):
        for tracker in self.trackers:
            tracker.predict()

    def addNewTrackers(self, detections):
        for detection in detections:
            tracker = Tracker(self.id, detection, self.nbImage)
            self.trackers.append(tracker)
            self.id = self.id + 1

    def updateMatchedTrackers(self, matched, detections):

        for m in matched:
            d = int(m[0])
            t = int(m[1])
            self.trackers[t].update(detections[d])

    def updateTrackers(self, detections):
        self.nonDetected = []
        self.predictTrackers()
        trackersPredictions = self.getPredictions()

        matched_idx, unmatchedTrackers, unmatchedDetections = self.hungarianAlgorithme.hungarian(self.trackers, detections)
        
        self.updateMatchedTrackers(matched_idx, detections)
        self.updateTrackersNotDetected(unmatchedTrackers)
        self.addNewTrackers(unmatchedDetections)
        self.nbImage = self.nbImage + 1
        
        return self.trackers, self.nonDetected
        """
        self.nonDetected = []
        if(len(self.trackers) == 0):
            self.addNewTrackers(detections)
        else:
            self.predictTrackers()
            self.associateTrackers(detections)
            #self.updateTrackersNotDetected()
        self.nbImage = self.nbImage + 1
        return self.trackers, self.nonDetected
        """

    def __init__(self, nbImage = 0):
        self.id = 1
        self.trackers = []
        self.nbImage = nbImage
        self.hungarianAlgorithme = HungarianAlgorithme()