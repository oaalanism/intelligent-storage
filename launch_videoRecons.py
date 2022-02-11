from classes.videoReconstruction import VideoReconstruction

"""
Summary:
This script launch video reconstruction of a set of depth data

Arguments
----------
VERSION : Int
    Version number of data
"""


VERSION = 71

DIRECTORY = "./output/v" + str(VERSION) + "/"

reconstruction = VideoReconstruction(DIRECTORY)
reconstruction.start()
reconstruction.showVideo()

resolution = [424, 240] 
nb_pixels_max = 5000
scope = [50, 3000]
minChange = 50

#reconstruction.saveVideo(resolution, nb_pixels_max, minChange)