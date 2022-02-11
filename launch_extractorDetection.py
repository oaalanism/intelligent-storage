from classes.extractorDetection import ExtractorDetection

VERSIONS = {
    1: 150,
    4: 30,
    5: 25,
    21: 25,
    23: 25,
    28: 25,
    30: 540,
    31: 20,
    32: 1680,
    33: 60,
    35: 10,
    41: 25,
    57: 25,
    58: 1020,
    60: 1745,
    69: 30,
    70: 25,
    71: 15
}

VERSION = 41

NB_REF_FRAME = VERSIONS[VERSION]

fps = 1/30
num_img = 0

extractorTrainingData = ExtractorDetection(VERSION, NB_REF_FRAME)
extractorTrainingData.start(fps, num_img)