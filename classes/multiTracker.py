import numpy as np

from classes.tracker import Tracker
from classes.hungarianAlgortihme import HungarianAlgorithme

class MultiTracker:
    """
    This object is a multy tracking that use Kalman Filter and Hungarian Algorithm 
     
    
     Parameters
     ----------
       nbImage: Int
            Image number where tracking begins
     """
    
    def __init__(self, nbImage = 0):
        self.id = 1
        self.trackers = []
        self.nbImage = nbImage
        self.hungarianAlgorithme = HungarianAlgorithme()
        
    def updateTrackerNotDetected(self, tracker):
        """
        Function to update tracker which has not been detected
        If tracker has not been detected in 15 seconds then is deleted
        Parameters
        ---------
        
            tracker: Tracker Object
                Tracker to update
        Returns
        ---------
        """
        tracker.timesUndectected = tracker.timesUndectected + 1
        if tracker.timesUndectected > 15:
            tracker.delete(self.nbImage)
            self.nonDetected.append(tracker)
            self.trackers.remove(tracker)

    def updateTrackersNotDetected(self, trackersNonDetected):
        """
            Function to loop in the list of trackers which have not been detected to update them
            Parameters
            ---------
                trackersNonDetected: Array of trackers objects
                    List of trackers objects which have not detected
            Returns
            ----------
        """
        for tracker in trackersNonDetected:
            self.updateTrackerNotDetected(tracker)

    def associateTrackers(self, detections):
        """
        Function that make association of trackers and detections
        Parameters
        ----------
            detections: Array
                Bounding boxes of detections in a frame
        Returns
        ---------
            trackers: Array
                List of tracker objects updated
            trackersNotAssociated: Array
                List of tracker objects that have not been associated
            detectionsNotAssociated: Array
                List of detections thath have not been associated
        """
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
        """
        Get predictions of Kalman Filters
        Paramaters
        ----------
        Returns
        ---------
            predictions : Array
                List of Kalman Filter predictions
        """
        predictions = []
        for tracker in self.trackers:
            predictions.append(tracker.KalmanFilter.X_)
        return predictions

    def predictTrackers(self):
        """
        Kalman filter tracker prediction feature released
        Parameters
        ---------
        Returns
        -------
        """
        for tracker in self.trackers:
            tracker.predict()

    def addNewTrackers(self, detections):
        """
        New trackers are created for those detectections that have not been detected
        
        Parameters
        ----------
        
        Returns
        -------
        """
        for detection in detections:
            tracker = Tracker(self.id, detection, self.nbImage)
            self.trackers.append(tracker)
            self.id = self.id + 1

    def updateMatchedTrackers(self, matched, detections):
        """
        Trackers matched with detections are update
        
        Parameters
        ----------
            matched: Array
                List of trackers objects that has matched with a detection
            detections: Array
                List of bounding boxes detections
        Returns
        -------
        
        """

        for m in matched:
            d = int(m[0])
            t = int(m[1])
            self.trackers[t].update(detections[d])

    def updateTrackers(self, detections):
        """
        This function launch others function to update trackers
        
        Parameters
        ----------
            detections: Array
                List of detections 
        Returns
        -------
            trackers: Array
                List of object trackers associated with detection
            nonDetected: Array
                List of trackers which has been associated in more of 15 seconds
        """
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

