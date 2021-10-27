from comparator import Comparator
import sys
"""

rawDataPath = './output/raw_data/'
algoPath = './output/algo/'
videoPath = './output/video/stream.avi'
"""

outputPath = sys.argv[1]

rawDataPath = outputPath + "raw_data/"
algoPath = outputPath + "algo/"
videoPath = outputPath + "video/"


comparator = Comparator(rawDataPath, algoPath, videoPath)
comparator.start()
comparator.algoReader.showVideo()
