import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


from classes.imageOutils import ImageOutils
from classes.outils import Outils

from scipy.signal import argrelextrema
from skimage.feature import greycomatrix, texture
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import shannon_entropy
from skimage import feature

class ExtractorFeatures:
    """
    This object extract features from bounding boxes detections
    Parameters
    ----------
    """
	
    def __init__(self):
        self.imageOutils = ImageOutils()

    def extractDepthFeatures(self, depth_frame):
    """
    This function extract depth vector features defined in the article "People re-identification using depth and intensity information from an overhead camera"
    
    The algorithm works according to the following steps :
    	- Group the pixels into 20 slides with a difference of 2cm. The first slide starts with the highest value of the frame.
	- The depth vector has 4 variables with the number of pixels in a group of 5 slides.
    Parameters
    ----------
    	depth_frame : Array
		Depth frame to extract this feature
    """
        highest_distance = np.amax(depth_frame)
        d = np.zeros(4)
        for i in range(4):
            d[i] = len(depth_frame[np.where(np.bitwise_and(highest_distance >= depth_frame, depth_frame >= highest_distance-90))])
            highest_distance = highest_distance - 90
        return d

    def extractEntropy(self, frame_gray):
    """
    This function extracts the entropy of a gray image created from a depth frame, this image is the head. 
    
    
    In a nutshell, the function extracts the cooccurrence matrix and calculates the entropy

    Parameters
    ----------
    	frame_gray : Array
		gray frame constructed from the depth image.
    Return
    ----------
    	ent : Float
		Entropy of a gray frame
    """

        glcm = greycomatrix(frame_gray, [1], [0], 256, True)
        coorelation = glcm[:,:,0,0]
        ent = shannon_entropy(glcm)
        #ent = entropy(coorelation, disk(5))

        return ent
    
    def localBinaryPatterns(self, depth_gray, numPoints, radius, eps=1e-7):

    """
    This function calculates the local binary patterns of a gray depth frame.
    LBPs are a texture descriptor, this function uses an extension of the original algorithm to consider more details in the image.
    To get this descriptor local_binary_pattern function is used
    
    For more information consult the next article : https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    	
    Parameters
    ----------
    	depth_gray : Array
		Gray head person frame
	numPoints : Int 
		Parameter of the local_binary_pattern function, this parameter indicates the number of pixels to be considered during the algorithm.
	radius : Int
		Parameter of the local_binary_pattern function, maximum radius of pixels to be calculated
    Return
    ----------
    	hist : Array
		Histogram of LBP values
    """

        #cv.imshow("head", depth_gray)
        #cv.waitKey(1000)
	glcm = greycomatrix(depth_gray, [1], [0], 256, True)
        lbp = feature.local_binary_pattern(glcm, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
        #plt.hist(hist)
        #plt.show()
        #hist = hist.astype("float")
		
        #hist /= (hist.sum() + eps)

       
        return hist



    def findCrosse(self, sigma, curvature, CSS):
	"""
   	Function that find crosse zero points in the curvature 
    	
    	Parameters
    	----------
		curvature: Array
			curve evolution
		CSS: Array
			the zero crossing from other old sigmas
    	Return
    	----------
    		CSS: Array
			the zero crossing updated
    	"""
        for i in range(0, curvature.size - 2):
            if (curvature[i] < 0.0 and curvature[i + 1] > 0.0) or (curvature[i] > 0.0 and curvature[i + 1] < 0.0):
                row = int(CSS.shape[0] - 10*sigma - 1)
                CSS[row, i] = 0
        
        return CSS

    def parametrizeCurvature(self, X, x0, Y, y0):
	"""
   	Function that parametrize contour curvature 
    	
    	Parameters
    	----------
    		X: Array
			X coordinates of cuvarture
		x0: Int
			x0 coordinate of center of the curvature
		Y: Array
			Y coordinates of cuvarture
		y0: Int
			y0 coordinate of center of the curvature
    	Return
    	----------
    		r: Array
	 		r values of the parametrization
		theta: Array
			theta values of the parametrization
    	"""
        theta_poly = np.arctan2(Y - y0, X - x0)
        r_poly = np.sqrt((X - x0)**2 + (Y - y0)**2)

        Z = np.polynomial.polynomial.polyfit(theta_poly, r_poly, 42)

        theta = np.linspace(0, 2*np.pi, 180)
        theta = np.where(theta > np.pi, (-1)*(2*np.pi - theta), theta)

        r = np.polynomial.polynomial.polyval(theta, Z)

        return r, theta

    def extractCurvatureScaleSpace(self, X, Y, x0, y0, shape, sigmaStart=0, sigmaEnd=0.6, sigmaStep=0.1, show=False):
    	"""
   	This function calculates Curvature Scale Space feature. This feature is inspired from the next articles :
	
	-----------------------https://vgg.fiit.stuba.sk/2013-04/css-%E2%80%93-curvature-scale-space-in-opencv/-------------------------------------
	-----------------------http://home.iitk.ac.in/~amit/courses/768/99/gunjan/#method-----------------------------------------------------------
   
   	This function follows the next steps :
		- Curvature of the contours is parametrized in the variables radius and theta. It is a simple transformation from Cartesian to polar coordinates.
		- Next, the Gaussian kernel is calculated
		- Curve evolution is computed by convolution of countour points with Gaussian Kernel
		- The derivates of the Gaussian kernel are computed to obtain 1st and 2nd derivation of contour points
		- 1st and 2nd derivation of contour points are calculated by convolution of coordinates with derivates of the Gaussian kernel.
		- The curvature is calculated and the zero crossings are found.
		- Then zero-croissing points are plotted 
		- Finally, the maximas of the zero-croissing are obtained
    	
    	Parameters
    	----------
    		X: Array
			X coordinates of cuvarture
		Y: Array
			Y coordinates of cuvarture
		x0: Int
			x0 coordinate of center of the curvature
		y0: Int
			y0 coordinate of center of the curvature
		shape: Array
			shape of depth frame
		sigmaStart: Float
			Value to start sigma value
		sigmaEnd: Float
			Value to end sigma value
		sigmaStep: Float
			Step of evolution of sigma
		show: Bool
			Variable indicating if the function shows the evolution of the curvature and the evolution of the zero crossing.
    	Return
    	----------
    		maximas: Array
	 		Maximas of non-zero curvature evolution
		CSS_image: Array
			Curvature Scale Space image of zero crossing curvature evolution
    	"""
        r,theta = self.parametrizeCurvature(X, x0, Y, y0)

        sigmas = np.linspace(sigmaStart, sigmaEnd, int(sigmaEnd/sigmaStep))
        CSS = np.ones((len(sigmas)+1, 500))*255

        for sigma in sigmas:
            X = x0 + r*np.cos(theta)
            Y = y0 + r*np.sin(theta)
            
            kernel = cv.getGaussianKernel(len(X), sigma, cv.CV_64FC1)
            G = cv.transpose(kernel)

            if show:
                ## See evolution of courve with kernel
                head = np.zeros(shape, np.uint8)
                Xsmooth = cv.filter2D(X, -1, kernel, borderType=cv.BORDER_DEFAULT)
                Ysmooth = cv.filter2D(Y, -1, kernel, borderType=cv.BORDER_DEFAULT)

                points = np.stack((Xsmooth, Ysmooth), -1).reshape((-1,1,2)).astype(np.int32)
                head = cv.drawContours(head, [points], -1, 255, 1)
                
                cv.imshow("sigma", head)
                cv.waitKey(100)

            dG = cv.Sobel(G, -1, 1, 0)
            ddG = cv.Sobel(G, -1, 2, 0)

            Xu = np.convolve(X, dG[0])
            Xuu = np.convolve(X, ddG[0])

            Yu = np.convolve(Y, dG[0])
            Yuu = np.convolve(Y, ddG[0])

            curvature = (Xu*Yuu - Xuu*Yu)
            
            CSS = self.findCrosse(sigma, curvature, CSS)
        
        
            
        points_crossings = np.where(CSS == 0)
        points_crossings = np.stack((CSS.shape[0] - points_crossings[0], points_crossings[1]), -1)
        points_crossings = np.array(sorted(points_crossings, key=lambda t:t[1]))
        #indices_maximas,_ = find_peaks(points_crossings[:,0], distance=20)
        indices_maximas = argrelextrema(points_crossings[:,0], np.greater)
        maximas = points_crossings[indices_maximas]
        maximas[:,0] = CSS.shape[0] - maximas[:,0]

        CSS_image = np.stack((CSS, CSS, CSS), -1)

        for x, y in zip(maximas[:,1], maximas[:,0]):
            CSS_image = cv.circle(CSS_image, (x,y), 1, (255, 0, 0), -1)

        return maximas, CSS_image


    def getArea(self, depth_frame_gray):
    	"""
   	This function calculates the area of the current segmented frame.
   
   	This function follows the following steps: 
   		- Extraction of the contours from the gray image.
		- With the contours the area is calculated with the contourArea function.
    	
    	Parameters
    	----------
    		depth_frame_gray : Matrix
			Segmented detection depth frame transformed into a gray image. 
    	Return
    	----------
    		area : Float
	 		Area of the segmented detection
    	"""
		
        min_val = np.amin(depth_frame_gray)
        max_val = np.amax(depth_frame_gray)
        cnt = self.imageOutils.extractContour(depth_frame_gray, min_val, max_val)
        area = 0 
        if len(cnt) > 0:
            area = cv.contourArea(cnt)
        return  area

    def extractMaxAndMinDistances(self, depth_frame):
        """
        Function to obtain maximum and minimum distance in a depth segmented frame
    	
        Parameters
        ----------
    		depth_frame : Array
			Current detection depth frame
    	Return
    	----------
    		min_distance : Int
	 		Minimum non-zero distance in the segmented depth frame.
	 	max_distance : Int
	 		Maximum distance in the segmented depth frame.

    	"""
        if(np.count_nonzero(depth_frame) > 0):
            max_distance = np.amax(depth_frame)
            min_distance = np.amin(depth_frame[np.where(depth_frame != 0)])
        else:
            max_distance = 0
            min_distance = 0
        return min_distance, max_distance


    def extractFeatures(self, depth_frame):
    """
    Main function that extracts all the features of a detection frame
    Before extracting the features, the detection is segmented into two parts: the head and the body, which are treated separately.
    	
    Parameters
    ----------
    	depth_gray : Array
		Gray segmented frame head or body
    Return
    ----------
    	 min_distance : Int
	 	Min distance different of zero in the segmented depth frame. For more information read description of extractMaxAndMinDistances function
	 max_distance : Int
	 	Max distance in the segemented depth frame. For more information read description of extractMaxAndMinDistances function
	 area: Int 
	 	Area of segmented depth frame. For more information read description of getArea function
	 maximas: Array
	 	Maximas coordinates in the CSS image. For more information read description of extractCurvatureScaleSpace function
	 entropy: Float
	 	Entropy of depth image. For more information read description of extractEntropy function
	 depth_vector : Array
	 	Depth vector of the depth frame. For more information read description of extractDepthFeatures function

    """
        depth_frame_gray_3D = self.imageOutils.getGrayFrame3D(depth_frame)
        cnt = self.imageOutils.extractContour(depth_frame_gray_3D)[0]
        approx = cv.approxPolyDP(cnt, 0.00001  * cv.arcLength(cnt, True), True)

        X = np.array([x[0] for x in approx[:,0]])
        Y = np.array([y[1] for y in approx[:,0]])

        x, y, w, h = cv.boundingRect(cnt)

        x0 = x + int(w/2)
        y0 = y + int(h/2)
        min_distance, max_distance = self.extractMaxAndMinDistances(depth_frame)
        area = self.getArea(cnt)
        
        maximas,CSS_image = self.extractCurvatureScaleSpace(X, Y, x0, y0, depth_frame_gray_3D.shape, show=False)
        bb = depth_frame_gray_3D[y:y+h, x:x+w]
        
        entropy = self.extractEntropy(bb)

        depth_vector = self.extractDepthFeatures(depth_frame, min_distance)
        
        return min_distance, max_distance, area, maximas, entropy, depth_vector
