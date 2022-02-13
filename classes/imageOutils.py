import fractions
import cv2 as cv
import numpy as np

from classes.colorizeDepthImage import ColorizeDepthImage
from classes.outils import Outils

from scipy.ndimage.measurements import label

class ImageOutils:
    """
    Image Outils object has a set of function to manipulate the images
   
    Parameters
    ----------
    """
    
    def __init__(self):
        self.colorizer = ColorizeDepthImage()
        
    def getGrayFrame(self, depth_frame):
        """
        This function transform depth frame into gray image
        Parameters
        ----------
            depth_frame : Array
                Depth frame
        Returns
        -------
            depth_frame_gray : Array
                Depth frame transformed in gray image
        """
        depth_frame_colorizer = self.colorizer.appplyColorization(depth_frame, 0, 3000)
    
        depth_frame_gray = cv.cvtColor(depth_frame_colorizer, cv.COLOR_RGB2GRAY)

        return depth_frame_gray

    def getGrayFrame3D(self, depth_frame):
        """
        Function to obtain a three-dimentional gray image. This image is used to draw bounding boxes
        Parameters
        ----------
            depth_frame : Array
                Depth frame
        Returns
        -------
            depth_frame_gray_3D : Array
                Gray depth image
        """
        depth_frame_gray = self.getGrayFrame(depth_frame)
        depth_frame_gray_3D = np.stack([depth_frame_gray, depth_frame_gray,depth_frame_gray], axis=-1)
        return depth_frame_gray_3D


    def transformRedToBlack(self, depth_frame_colorizer_):
        """
        this function is used to blacken background of the colored image background after background supression
        Parameters
        ----------
            depth_frame_colorizer_ : Array
                colored depth frame
        Returns
        -------
            depth_frame_colorizer_ : Array
                colored depth frame with black background
        """
        r = depth_frame_colorizer_[:,:,2]
        b = depth_frame_colorizer_[:,:,1]
        g = depth_frame_colorizer_[:,:,0]
        r = np.where(np.bitwise_and(r == 255, np.bitwise_and(b == 0, g == 0)), 0, r)
        depth_frame_colorizer_[:,:,2] = r

        return depth_frame_colorizer_

    def extractDepth(self, depth_boolean, depth_frame, pixel):
        """
        This function calculates the labeling of a depth chart to segment it.
        Parameters
        ----------
            depth_boolean : Array
                Depth indicates pixels to save
            depth_frame : Array
                Original depth frame
            pixel : Array
                Coordinates of highest value
        Returns
        -------
            depth_frame_ : Array
                Segmented depth frame
        """
        
        i = pixel[0]
        j = pixel[1]
        depth_frame_ = np.where(depth_boolean, depth_frame, 0)
        structure = np.ones((3, 3), dtype=np.float32)
        labeled, ncomponents = label(depth_boolean, structure)

        depth_frame_ = np.where(labeled == labeled[i][j], depth_frame, 0)
        #depth_frame_flatten = depth_frame_head.flatten()
        #var = np.var(depth_frame_flatten[np.nonzero(depth_frame_flatten)])
        """
        depth_frame_colorizer_ = ColorizeDepthImage.appplyColorization(depth_frame_, 0, 3000)
        depth_frame_colorizer_ = self.transformRedToBlack(depth_frame_colorizer_)
        r = depth_frame_colorizer_[:,:,2]
        b = depth_frame_colorizer_[:,:,1]
        g = depth_frame_colorizer_[:,:,0]
        r = np.where(np.bitwise_and(r == 255, np.bitwise_and(b == 0, g == 0)), 0, r)
        depth_frame_colorizer_[:,:,2] = r
        """
        return depth_frame_

    def segmentation(self, depth_frame):
        """
        This function segments depth frame of a detection to get head and body
        Parameters
        ----------
            depth_frame : Array
                Depth frame detection
        Returns
        -------
            depth_frame_head : Array
                Depth frame corresponding to head detection
            depth_frame_body : Array
                Depth frame corresponding to body detection
        """
        depth_frame = cv.medianBlur(depth_frame.astype(np.float32),3)
        hight = np.amax(depth_frame)
        highestPixel = np.where(depth_frame == hight)

        #distances = Outils.getDistances(depth_frame_smooth)
        #highestPixel = Outils.findHighestPixel(depth_frame_smooth, distances, 20)

        i = highestPixel[0][0]
        j = highestPixel[1][0]
        
        depth_boolean_head = (depth_frame<hight-200) & (depth_frame<=hight)
        depth_boolean_body = (hight-750 <= depth_frame) 

        #depth_frame_colorizer = ColorizeDepthImage.appplyColorization(depth_frame, 0, 3000)
        depth_frame_head = self.extractDepth(depth_boolean_head, depth_frame, [i,j])
        depth_frame_body = self.extractDepth(depth_boolean_body, depth_frame, [i,j])

        #depth_frame_gray = cv.cvtColor(depth_frame_colorizer, cv.COLOR_BGR2GRAY)
        #depth_frame_gray_head = cv.cvtColor(depth_frame_colorizer_head, cv.COLOR_BGR2GRAY)
        #depth_frame_gray_body = cv.cvtColor(depth_frame_colorizer_body, cv.COLOR_BGR2GRAY)
        depth_frame_body = abs(depth_frame_head - depth_frame_body)

        return depth_frame_head, depth_frame_body


    def extractContour(self, frame, min_val, max_val):
        """
        Function to extract contour of an image
        Parameters
        ----------
            frame: Array
                Depth frame image
            min_val: Int
                Minimum value in the depth frame
            max_val: Int
                Maximum value in the depth frame
        Returns
        ---------
            cnt: Array
                Contours of depth image
        """
        
        ret, thresh = cv.threshold(frame, 29, 255, 0)
        
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnt = []
        if(len(contours) > 0):
            cnt = contours[0]
        

        

        """
        img = np.zeros(frame.shape)
        img = cv.drawContours(img, cnt, -1, 255, -1)
        cv.imshow("img", img)
        cv.waitKey(100)
        """
        return cnt

    def extractBackground(self, reference_depth_frame, depth_frame):
        """
        This function removes background from an image using a frame reference
        
        Parameters
        ----------
            reference_depth_frame: Array
                Depth reference frame
            depth_frame : Array
                Depth frame to remove background
        Returns
        ---------
            depth_frame : Array
                Depth frame without beckground
        """
        depth_diference = np.abs(depth_frame - reference_depth_frame)
        depth_frame = np.where(depth_diference > 500, depth_frame, 0)

        depth_frame_coloriser = self.colorizer.appplyColorization(depth_frame, 0, 3000)
        depth_frame_gray = cv.cvtColor(depth_frame_coloriser, cv.COLOR_BGR2GRAY)
        mask = cv.threshold(depth_frame_gray, 100, 255, cv.THRESH_BINARY)[1]
        mask = 255 - mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv.BORDER_DEFAULT)
        mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
        mask = cv.medianBlur(mask, 45)
        mask = cv.medianBlur(mask, 45)
        depth_frame = np.where(mask == 0, depth_frame, 0)

        return depth_frame

    
