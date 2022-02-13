import cv2 as cv
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

from classes.colorizeDepthImage import ColorizeDepthImage
from classes.extractorFeatures import ExtractorFeatures
from classes.videoReconstruction import VideoReconstruction
from classes.imageOutils import ImageOutils

VERSIONS = {
    1: 150,
    4: 30,
    5: 25,
    21: 25,
    23: 25,
    28: 25,
    30: 540,
    31: 20,
    32: 1680,
    33: 60,
    35: 10,
    41: 25,
    57: 25,
    58: 1020,
    60: 1745,
    69: 30,
    70: 25,
    71: 15
}


class ExtractorTrainingData:
    """
    The training data extractor object extracts the data features from each detection and stores them in a csv file.
    Before extracting the features from the data, each detection must have the global identification of each person. These tags are added manually by the programmer.
    
    Feature extracted is:
        id: global id of each detection
        max_head: maximum head distance
        min_head: minimum head distance
        max_body: maximum body distance
        min_body: minimum body distance
        entropy: head entropy
        depth_vector_1: first component of the depth vector feature
        depth_vector_2: second component of the depth vector feature
        depth_vector_3: third component of the depth vector feature
        depth_vector_4: fourth component of the depth vector feature
        area_head: Head area
        area_body: Body area
        local_binary: head local binary
    
    Parameters
    ----------
    """
    def __init__(self) -> None:
        self.trainingDirectory = "./data/detection/"
        if(not(os.path.isdir(self.trainingDirectory))):
            print("Detections directory doesn't exist")
            sys.exit()
        else:
            
            self.getDetections()
            #self.extractFeaturesFromDetections()

    """
    def extractFeaturesFromDetections(self):
        print("Extracting Features...")

        idx = np.unique(self.data_file[:,2])

        versions = np.unique(self.data_file[:,0])
        
        colorizer = ColorizeDepthImage()
        extractorFeatures = ExtractorFeatures()
        imgOutils = ImageOutils()
        for version in versions:
            algoDirectory = "./output/v"+str(version)+"/algo/"
            algoReader = VideoReconstruction(algoDirectory)
            depth_frames = algoReader.start()
            detections = self.data_file[np.where(self.data_file[:,0] == version)]
            idx = np.unique(detections[:,3])
            for id in idx:
                id_detections = detections[np.where(detections[:,3] == id)]


            
            
            for detection in self.data_file[np.where(self.data_file[:,0] == version)]:
                num_img = detection[1]
                frame =  np.array(depth_frames[num_img].toarray(), dtype="float64")
                boundingBox = frame[detection[5]:detection[5]+detection[7], detection[4]:detection[4]+detection[6]]
                depth_frame_head, depth_frame_body = imgOutils.segmentation(boundingBox)

                depth_frame_gray_head = imgOutils.getGrayFrame3D(depth_frame_head)
                depth_frame_gray_body = imgOutils.getGrayFrame3D(depth_frame_body)
                cv.imshow("Head", depth_frame_gray_head)
                cv.imshow("Body", depth_frame_gray_body)
                cv.waitKey(1)
                #min_distance, max_distance, area, maximas, entropy, depth_vector = extractorFeatures.extractFeatures(boundingBox)
                #print(min_distance)

    """

    def getDetections(self):
        """
        This function reads the detections of each version to extract the features for re-identification which are stored in a csv file.
        
        Parameters
        ----------
        Returns
        -------
        
        """
        print("Extracting data...")
        f = open('./data/features.csv', 'w')
        writer = csv.writer(f)
        header = ['id', 'max_head', 'min_head', 'max_body', 'min_body', 'entropy', 'depth_vector_1', 'depth_vector_2', 'depth_vector_3', 'depth_vector_4', 'area_head', 'area_body', 'local_binary']
        data_features = []
        HCAMERA = 207
        self.data_file = []
        extractorFeatures = ExtractorFeatures()
        imgOutils = ImageOutils()
        
        for version in os.listdir(self.trainingDirectory):
            
            nbVersion = int(version.split("v")[1])
            nb_reference = VERSIONS[nbVersion]
            print(nbVersion)
            algoDirectory = "./output/"+version+"/algo/"
            algoReader = VideoReconstruction(algoDirectory)
            depth_frames = algoReader.start()
            reference_depth = np.array(depth_frames[nb_reference].toarray(), dtype="float64")
            
            file = open(self.trainingDirectory+version+"/"+version+".csv")
            
            df = pd.read_csv(self.trainingDirectory+version+"/"+version+".csv")
            df = df.astype("float")

            idx = np.unique(df["id_global"])

            for id_global in idx:

                detections_person = df[df["id_global"] == id_global ]

                passages = np.unique(detections_person["id_personne"])
                descripteurs = []
                for passage in passages:
                
                    detections_passage = detections_person[detections_person["id_personne"] == passage]
                    max_head = 0
                    min_head = HCAMERA*10

                    max_body = 0
                    min_body = HCAMERA*10

                    entropy_head = 0

                    found = False

                    for i in range(len(detections_passage)):
                        detection_passage = detections_passage.iloc[i]

                        num_image = int(detection_passage["num_image"])

                        cx = int(detection_passage["x"])
                        cy = int(detection_passage["y"])

                        if(abs(424/2 - cx) <= 40 and abs(240/2 - cy) <= 20  ):       

                            w = int(detection_passage["w"])
                            h = int(detection_passage["h"])
                            x = int(cx - round(w/2))
                            y = int(cy - round(h/2))

                            depth_frame = np.array(depth_frames[num_image].toarray(), dtype="float64")
                            depth_frame_wo_ba = imgOutils.extractBackground(reference_depth, depth_frame)
                            depth_frame_wo_ba_inv = np.where(depth_frame_wo_ba != 0, HCAMERA*10 - depth_frame_wo_ba, 0)

                            boundingBox = depth_frame_wo_ba_inv[y:y+h, x:x+w]

                            if(np.count_nonzero(boundingBox) > 0):

                                depth_frame_head, depth_frame_body = imgOutils.segmentation(boundingBox)

                                depth_frame_gray = imgOutils.getGrayFrame(depth_frame_wo_ba)
                                depth_frame_gray_head = imgOutils.getGrayFrame(np.where(depth_frame_head != 0, HCAMERA*10 - depth_frame_head, 0))
                                depth_frame_gray_body = imgOutils.getGrayFrame(np.where(depth_frame_body != 0, HCAMERA*10 - depth_frame_body, 0))

                                

                                

                                min_distance_head, max_distance_head = extractorFeatures.extractMaxAndMinDistances(depth_frame_head)
                                min_distance_body, max_distance_body =  extractorFeatures.extractMaxAndMinDistances(depth_frame_body)
                                
                                
                                if(max_head < max_distance_head):
                                    max_head = max_distance_head
                                    min_head = min_distance_head

                                    entropy_head = extractorFeatures.extractEntropy(depth_frame_gray_head)
                                    hist_head = extractorFeatures.localBinaryPatterns(depth_frame_gray_head, 24, 8)
                                    area_head = extractorFeatures.getArea(depth_frame_gray_head)

                                    max_body = max_distance_body
                                    min_body = min_distance_body
                                    area_body = extractorFeatures.getArea(depth_frame_gray_body)


                                    depth_vector = extractorFeatures.extractDepthFeatures(boundingBox)
                                
                                    cv.imshow("gray", depth_frame_gray)
                                    cv.imshow("head", depth_frame_gray_head)
                                    cv.imshow("body", depth_frame_gray_body)
                                    cv.waitKey(500)

                                    found = True
                                """
                                if(min_head > max_distance_head and max_distance_head > 0 and max_head - 350 <= max_distance_head):
                                    min_head = max_distance_head

                                if(max_body < max_distance_body and max_body < max_head - 350):
                                    max_body = max_distance_body
                                    area_body = extractorFeatures.getArea(depth_frame_gray_body)
                                if(min_body > max_distance_body and max_distance_body < max_body and max_distance_body > 0):
                                    min_body = max_distance_body
                                """

                            if(found):
                                data_features.append([detection_passage["id_global"], max_head,  min_head, max_body, min_body, entropy_head, depth_vector[0], depth_vector[1], depth_vector[2], depth_vector[3], area_head, area_body, sum(hist_head)/len(hist_head)])
                                print(id_global, passage, max_head, min_head, max_body, min_body, entropy_head)
                    """
                    idxmax = detections_passage["distance"].idxmax()
                    maxima_detection = df.iloc[idxmax]

                    
                    num_image = int(maxima_detection["num_image"])

                    w = int(maxima_detection["w"])
                    h = int(maxima_detection["h"])
                    x = int(maxima_detection["x"] - round(w/2))
                    y = int(maxima_detection["y"] - round(h/2))

                    depth_frame = np.array(depth_frames[num_image].toarray(), dtype="float64")
                    depth_frame_wo_ba = imgOutils.extractBackground(reference_depth, depth_frame)
                    depth_frame_wo_ba_inv = np.where(depth_frame_wo_ba != 0, HCAMERA*10 - depth_frame_wo_ba, 0)

                    boundingBox = depth_frame_wo_ba_inv[y:y+h, x:x+w]

                    depth_frame_head, depth_frame_body = imgOutils.segmentation(boundingBox)
                    depth_frame_gray_head = imgOutils.getGrayFrame(np.where(depth_frame_head != 0, HCAMERA*10 - depth_frame_head, 0))
                    depth_frame_gray_body = imgOutils.getGrayFrame(np.where(depth_frame_body != 0, HCAMERA*10 - depth_frame_body, 0))



                    min_distance_head, max_distances_head = extractorFeatures.extractMaxAndMinDistances(depth_frame_head)
                    min_distance_body, max_distances_body = extractorFeatures.extractMaxAndMinDistances(depth_frame_body)
                    area_head = extractorFeatures.getArea(depth_frame_gray_head)
                    area_body = extractorFeatures.getArea(depth_frame_gray_body)
                    entropy_head = extractorFeatures.extractEntropy(depth_frame_gray_head)
                    #print(id_global, entropy_head)
                    hist = extractorFeatures.localBinaryPatterns(depth_frame_gray_head, 24, 8)

                    depth_vector = extractorFeatures.extractDepthFeatures(boundingBox)
                    descripteurs = [maxima_detection["id_global"], max_distances_head,  min_distance_head, max_distances_body, min_distance_body, entropy_head, depth_vector[0], depth_vector[1], depth_vector[2], depth_vector[3], area_head, area_body, sum(hist)/len(hist)]
                    """
                    """
                    for e in hist:
                        descripteurs.append(e)
                    """

                    

                    #data_features.append(descripteurs)
            """
            csvreader = csv.reader(file)
            next(csvreader)
            detections = []
            for row in csvreader:
                r = []
                for cell in row:
                    r.append(int(float(cell)))
                detections.append(r)
            detections = np.array(detections)
            idx = np.unique(detections[:,2])
            
            
            for id in idx:
                idx_person = np.unique((detections[np.where(detections[:,2] == id)])[:,3])
                
                for id_person in idx_person:
                    detections_person = detections[np.where(detections[:,3] == id_person)]

                    max_heigh = max(detections_person[:,8])
                    maximas_detections = detections_person[np.where(max_heigh * 0.97 <= detections_person[:,8])]
                
                    for maximas_detection in maximas_detections:
                        nb_frame = maximas_detection[1]
                        depth_frame = np.array(depth_frames[nb_frame].toarray(), dtype="float64")
                        depth_frame_wo_ba = imgOutils.extractBackground(reference_depth, depth_frame)
                        depth_frame_wo_ba_inv = np.where(depth_frame_wo_ba != 0, HCAMERA*10 - depth_frame_wo_ba, 0)

                        highest_detections.append(maximas_detection)
                        w = maximas_detection[6]
                        h = maximas_detection[7]
                        x = maximas_detection[4] - round(w/2)
                        y = maximas_detection[5] - round(h/2)
                    
                        boundingBox = depth_frame_wo_ba_inv[y:y+h, x:x+w]

                        if(np.count_nonzero(boundingBox) > 0):
                            depth_frame_head, depth_frame_body = imgOutils.segmentation(boundingBox)
                            depth_frame_gray_head = imgOutils.getGrayFrame(np.where(depth_frame_head != 0, HCAMERA*10 - depth_frame_head, 0))
                            depth_frame_gray_body = imgOutils.getGrayFrame(np.where(depth_frame_body != 0, HCAMERA*10 - depth_frame_body, 0))

                            min_distance_head, max_distances_head = extractorFeatures.extractMaxAndMinDistances(depth_frame_head)
                            min_distance_body, max_distances_body = extractorFeatures.extractMaxAndMinDistances(depth_frame_body)
                            area_head = extractorFeatures.getArea(depth_frame_gray_head)
                            area_body = extractorFeatures.getArea(depth_frame_gray_body)
                            entropy_head = extractorFeatures.extractEntropy(depth_frame_gray_head)
                            depth_vector = extractorFeatures.extractDepthFeatures(boundingBox)

                            data_features.append([maximas_detection[2], max_distances_head,  min_distance_head, max_distances_body, min_distance_body, entropy_head, depth_vector[0], depth_vector[1], depth_vector[2], depth_vector[3], area_head, area_body])

        """
        """
                person_detection = detections[np.where(detections[:,2] == id)]
                highest = np.amax(person_detection[:,8])
                person_detections = person_detection[np.where(highest - person_detection[:,8] <= 50 )]
                for pd in person_detections:
                    highest_detections.append(pd)
            highest_detections = np.array(highest_detections)

            nb_frames = np.unique(highest_detections[:,1])

            for nb_frame in nb_frames:
                depth_frame = np.array(depth_frames[nb_frame].toarray(), dtype="float64")
                depth_frame_wo_ba = imgOutils.extractBackground(reference_depth, depth_frame)
                depth_frame_wo_ba_inv = np.where(depth_frame_wo_ba != 0, HCAMERA*10 - depth_frame_wo_ba, 0)

                detectionsInFrames = highest_detections[np.where(highest_detections[:,1] == nb_frame)]
                for detectionInFrame in detectionsInFrames:
                    w = detectionInFrame[6]
                    h = detectionInFrame[7]
                    x = detectionInFrame[4] - round(w/2)
                    y = detectionInFrame[5] - round(h/2)
                    
                    boundingBox = depth_frame_wo_ba_inv[y:y+h, x:x+w]
                    #detection_gray = imgOutils.getGrayFrame3D(np.where(boundingBox != 0, HCAMERA*10 - boundingBox, 0))
                    
                    if(np.count_nonzero(boundingBox) > 0):
                        depth_frame_head, depth_frame_body = imgOutils.segmentation(boundingBox)
                        depth_frame_gray_head = imgOutils.getGrayFrame(np.where(depth_frame_head != 0, HCAMERA*10 - depth_frame_head, 0))
                        depth_frame_gray_body = imgOutils.getGrayFrame(np.where(depth_frame_body != 0, HCAMERA*10 - depth_frame_body, 0))

                        min_distance_head, max_distances_head = extractorFeatures.extractMaxAndMinDistances(depth_frame_head)
                        min_distance_body, max_distances_body = extractorFeatures.extractMaxAndMinDistances(depth_frame_body)
                        area_head = extractorFeatures.getArea(depth_frame_gray_head)
                        area_body = extractorFeatures.getArea(depth_frame_gray_body)
                        entropy_head = extractorFeatures.extractEntropy(depth_frame_gray_head)
                        depth_vector = extractorFeatures.extractDepthFeatures(boundingBox)
                        data_features.append([detectionInFrame[2], max_distances_head,  min_distance_head, max_distances_body, min_distance_body, entropy_head, depth_vector[0], depth_vector[1], depth_vector[2], depth_vector[3], area_head, area_body])
                        #cv.imshow("detection", detection_gray)
                        #cv.imshow("head", depth_frame_gray_head)
                        #cv.imshow("body", depth_frame_gray_body)
                        #cv.waitKey(500)
                    
                    detection_gray = imgOutils.getGrayFrame3D(np.where(boundingBox != 0, HCAMERA*10 - boundingBox, 0))
                    depth_frame_head = np.where(depth_frame_head != 0, HCAMERA*10 - depth_frame_head, 0)
                    depth_frame_body = np.where(depth_frame_body != 0, HCAMERA*10 - depth_frame_body, 0)
                    depth_frame_gray_body = imgOutils.getGrayFrame3D(depth_frame_body)
                    cv.imshow("detection", detection_gray)
                    cv.imshow("head", depth_frame_gray_head)
                    cv.imshow("body", depth_frame_gray_body)
                    cv.waitKey(500)

                    """
        # write the header
        writer.writerow(header)

        # write multiple rows
        #data_features = sorted(data_features, key=lambda x:x[0])
        writer.writerows(data_features)
        self.data_file = np.array(self.data_file)
