import numpy as np

from classes.outils import Outils

class Detector:

    """
    Detector object implements algorithm of the article: "Robust People Detection Using Depth Information from an Overhead Time-of-Flight Camera"
    To detect body area of people in a scene
    
    This article is in the directory Biblio/Re-identification
    

    In a few words the algorithm cut each frames in regions, depending of camera parameters and 
    minimum distance to 
    """

    def getNeighborHood(self, currentRegion, v):
        regions = self.regions
        Nr = self.Nr
        Nc = self.Nc
        r = currentRegion[0]
        c = currentRegion[1]
        
        r1 = r - v
        r2 = r + v + 1

        c1 = c - v
        c2 = c + v + 1

        if(r1 < 0):
            r1 = 0

        if(r2 > Nr):
            r2 = Nr

        if(c1 < 0):
            c1 = 0
        
        if(c2 > Nc):
            c2 = Nc

        neigh_regions = np.zeros((Nr, Nc), dtype=bool)
        neigh_regions[r1:r2, c1:c2] = True
        neigh_regions[r,c] = False
        neigh_regions_values = regions[np.where(neigh_regions)]
        return neigh_regions_values

    def getMaximas(self):
        depth_frame = self.depth_frame
        #D = 3
        #self.D = 3
        D = self.D
        N = self.N
        M = self.M
        Nr = self.Nr
        Nc = self.Nc
        #Nr = N//D
        #Nc = M//D

        regions = np.zeros((Nr, Nc))

        for r in range(Nr):
            i1 = (r*D)
            i2 = (r+1)*D
            i1, i1 = self.outils.delimitate(0, N, i1, i1)
            i2, i2 = self.outils.delimitate(0, N, i1, i2)
            
            for c in range(Nc):

                j1 = (c*D)
                j2 = (c+1)*D 
                j1, j1 = self.outils.delimitate(0, M, j1, j1)
                j2, j2 = self.outils.delimitate(0, M, j2, j2)

                region = depth_frame[i1:i2, j1:j2]
                maxima = np.amax(region)
                #maxima = np.percentile(region, 0.95, axis=None)
                regions[r, c] = maxima
        
        #indeces = np.unravel_index(regions.argmax(), regions.shape)
        #i = indeces[0]*D
        #j = indeces[1]*D

        #depth_frame_gray = np.stack([depth_frame_gray, depth_frame_gray, depth_frame_gray], axis=-1)

        #depth_frame_gray = cv.circle(depth_frame_gray, (j, i),  5, (0, 255, 0), -1)
        #depth_frame_gray = cv.circle(depth_frame_gray, (j, i), 10, (0, 255, 0), 1)
        #cv.imshow("original",depth_frame_gray)
        #cv.waitKey(0)
        self.regions = regions
        #depth_frame_head, depth_frame_body = segmentation(depth_frame, [i, j, regions[indeces]])
        
        return regions

    def getMaximaNeighborhood(self, currentRegion, v):
        neigh_maximas = self.getNeighborHood(currentRegion, v)
        neigh_maxima = np.amax(neigh_maximas)

        return neigh_maxima

    def extractCandidates(self, v):
        Nr = self.Nr
        Nc = self.Nc
        regions = self.regions
        
        candidates = []
        regions_nonZero = np.nonzero(regions)

        indicesNonzero = np.stack((regions_nonZero[0], regions_nonZero[1]), axis=-1)

        for indices in indicesNonzero:
            current_maxima = regions[indices[0], indices[1]]
            neigh_maxima = self.getMaximaNeighborhood([indices[0],indices[1]], v)

            if(self.HPMIN <= current_maxima and current_maxima >=  neigh_maxima):
                candidates.append([indices[0], indices[1], current_maxima])
        
        """
        for i in range(Nr):
            for j in range(Nc):
                current_maxima = regions[i, j]
        """
                    
        self.candidates = np.array(candidates, dtype=np.uint64)
        return self.candidates

    def findNearby(self, candidates, v):
        i = 0
        new_candidates = []
        len_candidates = candidates.shape[0] 
        idx = []
        while i < len_candidates:

            distances = np.sqrt((candidates[:, 0] - candidates[i, 0])**2 + (candidates[:, 1] - candidates[i, 1])**2)

            nearbys = distances <= v
            nearbys[i] = False
            nearbys_region = candidates[nearbys]
            if(nearbys_region.shape[0] > 0 ):
                for index in np.where(nearbys):

                    candidates = np.delete(candidates, index, 0)
                len_candidates = candidates.shape[0]
                
                
                r = candidates[i, 0]
                c = candidates[i, 1]
                maxima = self.regions[r, c]
                for nearby in nearbys_region:
                    r = r + nearby[0]
                    c = c + nearby[1]
                    if(maxima < nearby[2]):
                        maxima = nearby[2]
                len_nearbys = nearbys_region.shape[0]+1 
                r = int(r/len_nearbys)
                c = int(c/len_nearbys)
                indices = np.unravel_index(np.argmax(self.depth_frame[r*self.D:(r+1)*self.D, c*self.D:(c+1)*self.D], axis=None), self.depth_frame[r*self.D:(r+1)*self.D, c*self.D:(c+1)*self.D].shape)
                
                new_candidates.append([r, c, maxima])
            else:
                new_candidates.append(candidates[i])
                r = int(candidates[i, 0])
                c = int(candidates[i, 1])
                indices = np.unravel_index(np.argmax(self.depth_frame[r*self.D:(r+1)*self.D, c*self.D:(c+1)*self.D], axis=None), self.depth_frame[r*self.D:(r+1)*self.D, c*self.D:(c+1)*self.D].shape)
            i_ = indices[0] + r*self.D
            j_ = indices[1] + c*self.D
            idx.append([j_, i_])
            i = i + 1

        return new_candidates, idx

    def getNeighborhoodDirection(self, region, v, direction):
        Nr = self.Nr
        Nc = self.Nc
        neigh = np.array([])
        if v > 0:
            regions = np.zeros((Nr, Nc), dtype=bool)
            
            r = region[0]
            c = region[1]
            
            i1 = r
            i2 = r + 1

            j1 = c 
            j2 = c + 1

            if(direction == 1):
                i1 = r - v
                i2 = r - v + 1
                j1 = c - (v - 1)
                j2 = c + v 
            elif(direction == 2):
                i1 = r - (v - 1)
                i2 = r + v
                j1 = c + v 
                j2 = c + v + 1
            elif(direction == 3):
                i1 = r + v
                i2 = r + v + 1
                j1 = c - (v - 1)
                j2 = c + v
            elif(direction == 4):
                i1 = r - (v - 1)
                i2 = r + v
                j1 = c - v
                j2 = c - v + 1
            elif(direction == 5):
                i1 = r - v
                i2 = r - v + 1
                j1 = c + v
                j2 = c + v + 1
            elif(direction == 6):
                i1 = r + v
                i2 = r + v + 1
                j1 = c + v
                j2 = c + v + 1
            elif(direction == 7):
                i1 = r + v
                i2 = r + v + 1
                j1 = c - v 
                j2 = c - v + 1
            elif(direction == 8):
                i1 = r - v
                i2 = r - v + 1
                j1 = c - v
                j2 = c - v + 1

            #i1, i2 = self.outils.delimitate(0, Nr, i1, i2)
            #j1, j2 = self.outils.delimitate(0, Nc, j1, j2)
            
            i1 = int(i1)
            i2 = int(i2)
            j1 = int(j1)
            j2 = int(j2)

            regions[i1:i2, j1:j2] = True

            neigh_reg = np.where(regions)

            neigh = np.stack([neigh_reg[0], neigh_reg[1]], axis=-1)

            next_neigh = self.getNeighborhoodDirection([r, c], v-1, direction)
            if next_neigh.size > 0:
                neigh = np.concatenate((neigh, next_neigh))
        
        return neigh

    def checkDecreasingHeightsAndAdd(self, ROI, ROI_, person, currentRegion, direction, h_interest):
        regions = self.regions
        Nr = self.Nr
        Nc = self.Nc
        r = int(person[0])
        c = int(person[1])
        hmax = regions[r, c]

        r_ = int(currentRegion[0])
        c_ = int(currentRegion[1])
        hmax_ = regions[r_, c_]

        delta_r = self.delta_r[direction - 1]
        delta_c = self.delta_c[direction - 1]

        r1 = int(r_ - 2*delta_r)
        c1 = int(c_ - 2*delta_c)
        r1, r1 = self.outils.delimitate(0, Nr, r1, r1)
        c1, c1 = self.outils.delimitate(0, Nc, c1, c1)
        hmax1 = regions[r1, c1]

        r2 = int(r_ - delta_r)
        c2 = int(c_ - delta_c)
        r2, r2 = self.outils.delimitate(0, Nr, r2, r2)
        c2, c2 = self.outils.delimitate(0, Nc, c2, c2)
        hmax2 = regions[r2, c2]

        r3 = int(r_ + delta_r)
        c3 = int(c_ + delta_c)
        r3, r3 = self.outils.delimitate(0, Nr, r3, r3)
        c3, c3 = self.outils.delimitate(0, Nc, c3, c3)
        hmax3 = regions[r3, c3]

        r4 = int(r_ + 2*delta_r)
        c4 = int(c_ + 2*delta_c)
        r4, r4 = self.outils.delimitate(0, Nr, r4, r4)
        c4, c4 = self.outils.delimitate(0, Nc, c4, c4)
        hmax4 = regions[r4, c4]

        if(hmax_ >= hmax - h_interest):
            R = [r_, c_, r_+1, c_+1]
            #if(((hmax3 <= hmax_ and hmax_ >= hmax2) or (hmax2 >= hmax_ and hmax_ >= hmax3) or (hmax3 >= hmax_ and hmax_ >= hmax2))):
            if(hmax2 >= hmax_ and hmax_ >= hmax3):
                ROI = np.concatenate((np.array([currentRegion]), ROI), axis=0)
                ROI_ = np.concatenate((np.array([R]), ROI_), axis=0)
                return True, ROI, ROI_
            else:
                currentRegion = currentRegion.astype("float32")

                if(delta_r < 0):
                    R[0] = r_ + 1/2
                elif(0 < delta_r):
                    R[2] = r_ + 1/2

                if(delta_c < 0):
                    R[1] = c_ + 1/2
                elif(0 < delta_c):
                    R[3] = c_ + 1/2
                
                ROI = np.concatenate((np.array([currentRegion]), ROI), axis=0)
                ROI_ = np.concatenate((np.array([R]), ROI_), axis=0)
                return False, ROI, ROI_
        else:
            return False, ROI, ROI_

    def getMaxNeighDirection(self, region, direction, v):
        regions = self.regions
        neighDirections = self.getNeighborhoodDirection(region, v, direction)
        neighDirectionsValues = regions[neighDirections]

        maxNeighDirection = np.amax(neighDirectionsValues)

        return maxNeighDirection
    
    def findNeighBody(self, ROI, ROI_, person, v, direction, h_Interest):
        h_max = person[2]
        regions = self.regions
        D = self.D
    
        for v_ in range(1, v):
            neighDirections = self.getNeighborhoodDirection(person, v_, direction)
            I = self.outils.intersectionRegions(ROI, neighDirections)
            
            decreasings = 0
            for neighDirectionRegion in neighDirections:

                r = neighDirectionRegion[0]
                c = neighDirectionRegion[1]
                a = r + 1
                b = c + 1

                i1 = int(r*D)
                i2 = int((r+1)*D)
                j1 = int(c*D)
                j2 = int((c+1)*D)

                i1, i2 = self.outils.delimitate(0, self.N, i1, i2)
                j1, j2 = self.outils.delimitate(0, self.M, j1, j2)

                #depth_frame_gray = cv.rectangle(depth_frame_gray, (j1, i1), (j2, i2), (0, 255, 0), 1)

                neighDirectionsValue = regions[neighDirectionRegion[0], neighDirectionRegion[1]]
                #cv.imshow("ROIS", depth_frame_gray)
                #cv.waitKey(0)
                if(neighDirectionsValue < h_max + h_Interest ):
                    #person = np.array([person[0], person[1]])
                    
                    if( 1 <= direction and direction <= 4) and len(I) >= v_ - 1:
                        decreasing, ROI, ROI_ = self.checkDecreasingHeightsAndAdd(ROI, ROI_, person, neighDirectionRegion, direction, h_Interest)
                        if(not(decreasing)):
                            decreasings = decreasings + 1
                    
                    elif( 5 <= direction and direction <= 8) and len(I) >= v_ - 1:
                        decreasing, ROI, ROI_ = self.checkDecreasingHeightsAndAdd(ROI, ROI_, person, neighDirectionRegion, direction, h_Interest)
                        if(not(decreasing)):
                            decreasings = decreasings + 1
            
            if(decreasings >= len(neighDirections)/2 ):
                return ROI, ROI_
        return ROI, ROI_

    def extractROI(self, person, h_Interest, v):
        
        r = person[0]
        c = person[1]
        
        ROI = np.array([[r, c]])
        ROI_ =  np.array([[r, c, r+1, c+1]])
        for direction in range(1, 9):
            ROI, ROI_ = self.findNeighBody(ROI, ROI_, person, v, direction, h_Interest)
        return ROI, ROI_

    def extractROIS(self, persons, v_ROIS, h_Interest):
        ROIS = []
        ROIS_ = []
        for person in persons:
            ROI, ROI_ = self.extractROI(person, h_Interest, v_ROIS)
            ROIS.append(ROI)
            ROIS_.append(ROI_)
        ROIS = np.array(ROIS)
        ROIS_ = np.array(ROIS_)
        return ROIS_

    def findBoundingBoxes(self, ROIS_people):
        BBs = []
        D = self.D
        for ROIS_person in ROIS_people:
            r = np.amin(ROIS_person[:,0])
            c = np.amin(ROIS_person[:,1])
            a = np.amax(ROIS_person[:,2])
            b = np.amax(ROIS_person[:,3])

            person = self.depth_frame[round(r*D):round(a*D), round(c*D):round(b*D)]
            idx = np.argwhere(person != 0)
            i1 = np.amin(idx[:,0])
            i2 = np.amax(idx[:,0])

            j1 = np.amin(idx[:,1])
            j2 = np.amax(idx[:,1])

            
            w = int(j2 - j1) + 1
            h = int(i2 - i1) + 1
            y = round(r*D)+int(i1) + round(h/2)
            x = round(c*D)+int(j1) + round(w/2)
            if(w*h >= D*D):
            
                BBs.append([x, y, w, h])
                

        return BBs

    def findSameBB(self, BBs):
        
        for i in range(len(BBs)-1):
            if(i < len(BBs)-1):
                bb1 = BBs[i]
                for j in reversed(range(1, len(BBs))):
                    
                    bb2 = BBs[j]
                    iou = self.outils.IOU(bb1, bb2)        
                    if(iou >= 0.1):
                        x1 = bb1[0]
                        y1 = bb1[1]
                        w1 = bb1[2]
                        h1 = bb1[3]
                        
                        x2 = bb2[0]
                        y2 = bb2[1]
                        w2 = bb2[2]
                        h2 = bb2[3]

                        if(x2 < x1):
                            bb1[0] = x2
                        if(y2 < y1):
                            bb1[1] = y2
                        
                        if(x1+w1 < x2+w2):
                            bb1[2] = w2
                        if(y1+h1 < y2+h2):
                            bb1[3] = h2

                        BBs.pop(j)
        return BBs
            

    def peopleDetection(self, depth_frame, v_Candidates = 2, v_Nearbys = 3, v_ROIS = 4, h_Interest = 400):
        D = self.D
        depth_frame = np.where(depth_frame != 0, self.HCAMERA*10 - depth_frame, 0)
        #depth_frame = cv.medianBlur(depth_frame.astype(np.float32), 15)
        
        #depth_frame = self.HCAMERA*10 - depth_frame
        
        self.N = depth_frame.shape[0]
        self.M = depth_frame.shape[1]
        self.Nr = round(self.N/D)
        self.Nc = round(self.M/D)
        self.depth_frame = depth_frame[0:self.Nr*D, 0:self.Nc*D]

        


        regions = self.getMaximas()
        candidates = self.extractCandidates( v_Candidates)
        persons, centroids = self.findNearby(candidates, v_Nearbys)

        ROIS = self.extractROIS(persons, v_ROIS, h_Interest)
        Bbs = self.findBoundingBoxes(ROIS)
        BBs = self.findSameBB(Bbs)

        return BBs, persons, ROIS, centroids

    def __init__(self, FOCAL, A, L, HCAMERA, HPMIN):
        self.HCAMERA = HCAMERA
        self.HPMIN = HPMIN
        self.D = round((FOCAL/A) * (L/(HCAMERA - HPMIN)))
        print(self.D )
        #self.D = 10
        self.delta_r = np.array([-1, 0, 1, 0, -1, 1, 1, -1])
        self.delta_c = np.array([0, 1, 0, -1, 1, 1, -1, -1])
        self.outils = Outils()