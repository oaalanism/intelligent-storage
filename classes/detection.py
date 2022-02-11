class Detection:

    def getBoundingBox(self):
        return self.boundingBox

    def updateBoundingBox(self, boundingBox):
        self.boundingBox = boundingBox

    def __init__(self, id=None, boundingBox=None, timeDetected = None, imageDetected = None):
        self.id = id
        self.boundingBox = boundingBox
        self.detected = False
        self.timesNotDetected = 0
        self.timeDetected = timeDetected
        self.imageDetected = imageDetected