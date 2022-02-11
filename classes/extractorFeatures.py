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


    def extractDepthFeatures(self, depth_frame):
        highest_distance = np.amax(depth_frame)
        d = np.zeros(4)
        for i in range(4):
            d[i] = len(depth_frame[np.where(np.bitwise_and(highest_distance >= depth_frame, depth_frame >= highest_distance-90))])
            highest_distance = highest_distance - 90
        return d

    def extractEntropy(self, frame_gray):

        glcm = greycomatrix(frame_gray, [1], [0], 256, True)
        coorelation = glcm[:,:,0,0]
        ent = shannon_entropy(glcm)
        #ent = entropy(coorelation, disk(5))

        return ent
    
    def localBinaryPatterns(self, depth_gray, numPoints, radius, eps=1e-7):

        #cv.imshow("head", depth_gray)
        #cv.waitKey(1000)
        lbp = feature.local_binary_pattern(depth_gray, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
        #plt.hist(hist)
        #plt.show()
        #hist = hist.astype("float")
		
        #hist /= (hist.sum() + eps)

       
        return hist



    def findCrosse(self, sigma, curvature, CSS):
        for i in range(0, curvature.size - 2):
            if (curvature[i] < 0.0 and curvature[i + 1] > 0.0) or (curvature[i] > 0.0 and curvature[i + 1] < 0.0):
                row = int(CSS.shape[0] - 10*sigma - 1)
                CSS[row, i] = 0
        
        return CSS

    def parametrizeCurvature(self, X, x0, Y, y0):
        theta_poly = np.arctan2(Y - y0, X - x0)
        r_poly = np.sqrt((X - x0)**2 + (Y - y0)**2)

        Z = np.polynomial.polynomial.polyfit(theta_poly, r_poly, 42)

        theta = np.linspace(0, 2*np.pi, 180)
        theta = np.where(theta > np.pi, (-1)*(2*np.pi - theta), theta)

        r = np.polynomial.polynomial.polyval(theta, Z)

        return r, theta

    def extractCurvatureScaleSpace(self, X, Y, x0, y0, shape, sigmaStart=0, sigmaEnd=0.6, sigmaStep=0.1, show=False):
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
        min_val = np.amin(depth_frame_gray)
        max_val = np.amax(depth_frame_gray)
        cnt = self.imageOutils.extractContour(depth_frame_gray, min_val, max_val)
        area = 0 
        if len(cnt) > 0:
            area = cv.contourArea(cnt)
        return  area

    def extractMaxAndMinDistances(self, depth_frame):
        """
        outils = Outils()
        distancesMinorMajor = outils.getDistances(depth_frame)
        highestPixel = outils.findHighestPixel(depth_frame, distancesMinorMajor, 20)

        distancesMajorMinor = outils.getDistances(depth_frame, True)
        minustPixel = outils.findHighestPixel(depth_frame, distancesMajorMinor, 20)

        min_distance = minustPixel[2]
        max_distances = highestPixel[2]
        """
        if(np.count_nonzero(depth_frame) > 0):
            max_distance = np.amax(depth_frame)
            min_distance = np.amin(depth_frame[np.where(depth_frame != 0)])
        else:
            max_distance = 0
            min_distance = 0
        return min_distance, max_distance


    def extractFeatures(self, depth_frame):
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
    def __init__(self):
        self.imageOutils = ImageOutils()