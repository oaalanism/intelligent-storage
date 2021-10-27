from classes.comparator import Comparator
import sys
"""

rawDataPath = './output/raw_data/'
algoPath = './output/algo/'
videoPath = './output/video/stream.avi'
"""

outputPath = None
if len(sys.argv) > 1:
    outputPath = sys.argv[1]

comparator = Comparator(outputPath)
comparator.start()
comparator.algoReader.showVideo()
