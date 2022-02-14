import numpy as np

class Outils:
    """
        Outils Object group a geat number of functions that are used in different objects
        Paramaters
        ----------
    """
    def __init__(self) -> None:
        pass

    def mean_neigh(self, frame, row, column):
        """
            Mean of distances in a pixel region
            
            Parameters
            ----------
                frame: Array
                    Depth frame
                row: Array
                    Coordinates of rows region
                column: Array
                    Coordinates of columns region
        """
        mean = 0

        row_ll = row - 1
        if(row_ll < 0): 
            row_ll = row

        row_ul = row + 2
        if(row_ul > frame.shape[0]):
            row_ul = row

        column_ll = column - 1
        if column_ll < 0:
            column_ll = column

        column_ul = column+2

        if(column_ul > frame.shape[1]):
            column_ul = column

        for i in range(row_ll, row_ul):
            for j in range(column_ll, column_ul):
                if i != row or j != column:
                    mean = mean +  frame[i,j]

        mean = mean/8

        return mean

    def findHighestPixel(self, depth_frame, distances, error=20):
        """
            Obtain the maximum value within a pixel region 
            Parameters
            ----------
                depth_frame: Array
                    Depth frame
                distances: Array
                    set of distances with pixel coordinates
        """
        highestPixel = None
        for distance in distances:
            if(distance[2] != 0):
                mean =  self.mean_neigh(depth_frame, distance[0], distance[1])
                if abs(mean - distance[2]) < error:
                    highestPixel = distance
                    break

        return highestPixel

    def getDistances(self, depth_frame, reverse=False):
        """
            Get distance of a depth frames with the corresponding coordinates 
            
            Parameters
            ----------
                depth_frame: Array
                    Depth frame
                    
                reverse: Boolean
                    True to sort distance from highest to lowest or vice versa
        """
        points = [[row, column, depth_frame[row, column]] for row in range(depth_frame.shape[0]) for column in range(depth_frame.shape[1])] 
        
        distances = sorted(points, key= lambda t:t[2], reverse=reverse)
        return distances

    def intersection(self, bb1, bb2):
        """
            Function that calculates intersection zone between two bounding boxes
            
            Paramaters
            ----------
                bb1: Array
                    First bounding box
                bb2: Array
                    Second bounding box
            Returns
            -------
                intersection: Float
                    Intersection zone            
        """

        xA = max(bb1[0], bb2[0])
        yA = max(bb1[1], bb2[1])
        xB = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
        yB = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

        return max(0, xB - xA + 1) * max(0, yB - yA + 1)

    def union(self, bb1, bb2, area_I):
        """
            Function that calculates union zone between two bounding boxes
            
            Paramaters
            ----------
                bb1: Array
                    First bounding box
                bb2: Array
                    Second bounding box
                area_I: Float
                    Zone intersection of the two bounding boxes
            Returns
            -------
                union: Float
                    Union zone
                    
        """
        w1 = bb1[2]
        h1 = bb1[3]
        
        w2 = bb2[2]
        h2 = bb2[3]

        area = w1*h1 + w2*h2 - area_I

        return area
        

    def IOU(self, bb1, bb2):
        """
            Calculate IOU between two bounding boxes
            A bounding box is an array with next values:
                x center coordinate 
                y center coordinate
                width of bounding box
                height of bounding box
                
            Parameters
            ----------
            bb1: Array
                First Bounding Box 
            bb2: Array
                Second Bounding Box
        """
        inter = self.intersection(bb1, bb2)

        uni = self.union(bb1, bb2, inter)
        
        return inter/uni

    def intersectionRegions(self, arr1, arr2):
        """
            Get intersection regions 
            
            Parameters
            ----------
                arr1: Array
                    Set of regions
                arr2: Array
                    Set of regions
            Returns
            -------
                Intersected Regions : Array
                    
        """
        arr1 = arr1.astype('float64')
        arr2 = arr2.astype('float64')
        arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
        arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
        intersected = np.intersect1d(arr1_view, arr2_view)
        return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

    def delimitate(self, limitInf, limitSup, varInf, varSup):
        """
            Function to delimitate two values between a limit
            
            Parameters
            ----------
                limitInf: Float
                    Inferior limit
                limitSup: Float
                    Superior limit                    
                varInf: Float
                    Inferior variable to delimitate
                varSup: Float
                    Superior variable to delimitate
        """

        if(varInf < limitInf):
            varInf = limitInf
        if(limitSup <= varSup):
            varSup = limitSup - 1

        return int(varInf), int(varSup)

