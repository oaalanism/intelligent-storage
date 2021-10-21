from comparator import Comparator
import sys
"""
rawDataPath = sys.argv[1]
algoPath = sys.argv[2]
videoPath = sys.argv[3]
"""
rawDataPath = './output/raw_data/'
algoPath = './output/algo/'
videoPath = './output/video/stream.avi'



comparator = Comparator(rawDataPath, algoPath, videoPath)
comparator.start()
