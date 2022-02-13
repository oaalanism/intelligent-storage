class Detection:
    """
    Representation of a Object
    Parameters
    ----------
        id : Int
            Identifier in the current video

        boundingBox : Array
            Array with bounding box information :
                index 0 : x coordinate center of bounding box
                index 1 : y coordinate center of bounding box
                index 2 : width of bounding box
                index 3 : height of bounding box 
    """
    
    def __init__(self, id=None, boundingBox=None, timeDetected = None, imageDetected = None):
        
        self.id = id
        self.boundingBox = boundingBox
        self.detected = False
        self.timesNotDetected = 0
        self.timeDetected = timeDetected
        self.imageDetected = imageDetected

    def getBoundingBox(self):
        """
        Function to get bounding box
        Parameters
        ----------
        Returns
        -------
            boundingBox : Array
                Bounding box of object
        """
        
        return self.boundingBox

    def updateBoundingBox(self, boundingBox):
        """
        Function to set bounding box
        Parameters
        ----------
        Returns
        -------
            boundingBox : Array
        """
        self.boundingBox = boundingBox

    
