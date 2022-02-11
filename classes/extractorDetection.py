import csv
import cv2 as cv
import numpy as np
import os

from classes.colorizeDepthImage import ColorizeDepthImage
from classes.detector import Detector
from classes.imageOutils import ImageOutils
from classes.multiTracker import MultiTracker
from classes.outils import Outils
from classes.videoReconstruction import VideoReconstruction

class ExtractorDetection:
    """
    Object Extractor extract detection from a set of depth data to save each one in a csv file with path ./data/detection/vXX/vXX.csv
    And a video of depth grey frame with detections

    
    The data saved is :
        Number of version
        Number of image
        Person id
        x coordinate center of the bounding box detection
        y coordinate center of the bounding box detection 
        width of bounding box
        heigt of bounding box
        maximum distance of detection
    

    Parameters
    ----------
        version : Int
            Number of data version to detect people
        nb_ref_frame : Int
            Reference number of frame to extract background
    """

    def __init__(self, version, nb_ref_frame):
        self.VERSION = version
        self.NB_REF_FRAME = nb_ref_frame
        self.DIRECTORY = "./output/v" + str(self.VERSION) + "/"
        self.ALGO_DIRECTORY = self.DIRECTORY + "algo/"
        self.traingingDirectory = "./training/v"+str(self.VERSION)+"/"

        FOCAL = 1.93/1000
        A = 9/1000000

        L = 15 #cm
        HCAMERA = 207 #cm
        HPMIN = 50 #cm
        self.algoReader = VideoReconstruction(self.ALGO_DIRECTORY)
        self.colorizer = ColorizeDepthImage()
        self.detector = Detector(FOCAL, A, L, HCAMERA, HPMIN)
        self.imageOutils = ImageOutils()
        self.outils = Outils()
        self.multiTracker = MultiTracker()
        

    def saveDataPassengers(self, header, data):
        """
        Function to save data detection
        Parameters
        ----------
        header : Array
            Header of data to store in the csv file

        data : Array
            Data detection

        Raises
        ------

        Returns
        -------
        
        """
        with open(self.traingingDirectory+'./v'+str(self.VERSION)+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            writer.writerows(data)

    def stopSaveVideo(self):
        """
        Function to release video with detections
        Parameters
        ----------

        Raises
        ------

        Returns
        -------
        
        """
        self.videoWriter.release()

    def saveFrame(self, frame):
        """
        Function to add depth frames into video 
        Parameters
        ----------

        Raises
        ------

        Returns
        -------
        
        """
        self.videoWriter.write(frame)

    def createFrame(self, depth_frame, passengers, second, num_img):
        """
        Function to draw bounding boxes into grey depth frame
        Parameters
        ----------
            depth_frame : Array
                Depth data

            passengers : Array
                Array with bounding boxes for each detection

            second : Int
                Current time of video

            num_img : Int
                Number of image

        Raises
        ------

        Returns
        -------

            depth_frame_gray_3D : Array
                Gray depth frame with detections of bounding boxes  
        
        """
        depth_frame_gray_3D = self.imageOutils.getGrayFrame3D(depth_frame)
        depth_frame_gray_3D = self.putFrameInformation(depth_frame_gray_3D, second, num_img)
        depth_frame_gray_3D = self.putBoundingBoxes(depth_frame_gray_3D, passengers)

        return depth_frame_gray_3D

    def getPassengers(self, depth_frame, data, num_img):
        """
        Function to get detections and update trackers

        Parameters
        ----------
            depth_frame : Array
                Depth data

            data : Array
                Data detection

            num_img : Int
                Number of image

        Raises
        ------

        Returns
        -------
            passengers : Array
                List of trackers updated Bounding Boxes

            persons : Array
                New detections

            ROIS : Array 
                Regions qui appartient à chaque detection

            centroids : Array
                Centrois pour chaque detection
        """
        detections, persons, ROIS, centroids = self.detector.peopleDetection(depth_frame, 3, 8, 5, 350)
        passengers, passengersNonDetected = self.multiTracker.updateTrackers(detections)
        data = self.savePassengersNotDetected(data, passengers, num_img)

        
        return passengers, persons, ROIS, centroids

    def savePassengersNotDetected(self, data, passengersNonDetected, num_img):

        """
        Function to store the information of passengers who have left the scene.

        Parameters
        ----------

            passengersNonDetected : Array
                Array with the list of passengers not detected

            data : Array
                Data detection

            num_img : Int
                Number of image

        Raises
        ------

        Returns
        -------
            data : Array
                Updated data 
        
        """
        for passenger in passengersNonDetected:
            distance = np.amax(self.detector.depth_frame[passenger.boundingBox[1]:passenger.boundingBox[1]+passenger.boundingBox[3], passenger.boundingBox[0]:passenger.boundingBox[0]+passenger.boundingBox[2]])
            data.append([self.VERSION, num_img, 0, passenger.id, passenger.boundingBox[0], passenger.boundingBox[1], passenger.boundingBox[2], passenger.boundingBox[3], distance])
        return data

    def putTextId(self, depth_frame_gray_3D, coord, id):

        """
        Put id of detection in frame

        Parameters
        ----------

            depth_frame_gray_3D : Array
                3D grey image of depth frame 

            coord : Array
                Coordinates of image to draw id

            id : Int
                Id of detection

        Raises
        ------

        Returns
        -------
            depth_frame_gray_3D : Array
                3D grey image of depth frame with id of detection 
        """

        txt_id = "ID: "+str(id)
        depth_frame_gray_3D = cv.putText(depth_frame_gray_3D, txt_id, coord, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

        return depth_frame_gray_3D

    def putBoundingBoxes(self, depth_frame_gray_3D, passengers):
        """
        Function to draw bounding boxes in grey depth frame 

        Parameters
        ----------

            depth_frame_gray_3D : Array
                3D grey image of depth frame 

            passenger : Array 
                Bounding Box of detection

        Raises
        ------

        Returns
        -------
            depth_frame_gray_3D : Array
                3D grey image of depth frame with bounding box
        """
        for passenger in passengers:
            
            box = passenger.boundingBox

            w = box[2]
            h = box[3]
            x = box[0] - round(w/2)
            y = box[1] - round(h/2)

            depth_frame_gray_3D = cv.rectangle(depth_frame_gray_3D, (x, y), (x+w, y+h), (0, 255, 0), 3)
            depth_frame_gray_3D = cv.rectangle(depth_frame_gray_3D, (x, y-20), (x+55, y), (0, 255, 0), -1)
            
            depth_frame_gray_3D = self.putTextId(depth_frame_gray_3D, (x+int(w/2), y+int(h/2)), passenger.id)

        return depth_frame_gray_3D

    def putFrameInformation(self, depth_frame_gray_3D, second, num_img):

        """
        Function to draw video information like second and number of image

        Parameters
        ----------

            depth_frame_gray_3D : Array
                3D grey image of depth frame 

            second : Int 
                Current time
            
            num_img : Int
                Image number

        Raises
        ------

        Returns
        -------
            depth_frame_gray_3D : Array
                3D grey image of depth frame with current information
        """

        txt_currentDetection = "Time: " + str(round(second, 2)) + " s"
        depth_frame_gray_3D = cv.putText(depth_frame_gray_3D, txt_currentDetection, (0, 210), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        txt_currentDetection = "Image: " + str(num_img)
        depth_frame_gray_3D = cv.putText(depth_frame_gray_3D, txt_currentDetection, (0, 225), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        return depth_frame_gray_3D

    def start(self, fps, num_img):

        """
        Function to start detection in the dept data

        Parameters
        ----------

            fps : Float
                fps of camera
            
            num_img : Int
                Image number

        Raises
        ------

        Returns
        -------
        """

        if(not os.path.exists(self.traingingDirectory)):
            os.mkdir(self.traingingDirectory)
        
        videoPath = self.traingingDirectory + "v"+str(self.VERSION)+".avi"
        self.videoWriter = cv.VideoWriter(videoPath, cv.VideoWriter_fourcc(*'XVID'), 20.0, (424, 240))

        depth_frames = self.algoReader.start()
        reference_depth = np.array(depth_frames[self.NB_REF_FRAME].toarray(), dtype="float64")

        second = num_img*fps
        data = []
        header = ['video', 'num_image', 'id_global', 'id_personne', 'x', 'y', 'w', 'h', 'distance']

        for depth_frame in depth_frames[num_img:]:
            
            depth_frame = np.array(depth_frame.toarray(), dtype="float64")
            depth_frame_wo_ba = self.imageOutils.extractBackground(reference_depth, depth_frame)

            self.depth_frame = depth_frame_wo_ba          
            passengers, persons, ROIS, centroids = self.getPassengers(depth_frame_wo_ba, data, num_img)
            depth_frame_gray_3D = self.createFrame(depth_frame, passengers, second, num_img)

            self.saveFrame(depth_frame_gray_3D)

            cv.imshow("depth frame gray", depth_frame_gray_3D)
            key  = cv.waitKey(1)

            if key == ord('q'):
                return False
            if key == ord('p'):
                cv.waitKey(-1)

            second = second + fps
            num_img = num_img + 1
        
        self.stopSaveVideo()
        self.saveDataPassengers(header, data)

        
    