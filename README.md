# intelligent-storage, detection and feature extraction

This is the re-identification project that has the goal to associate people using features obtained with a depth camera D455 of Intel RealSense.

##intelligent-storage

First store depth frame data : 

```
python3 launch_stream.py
```

The program convert depth data into three representations : 

* Raw Data -> ./ output/v{*last version*} / raw_data /
* Sparse matrix -> ./output/v{*last version*} / algo /
* RGB Video -> ./ output/v{*last version*} / video /

## Detections

Then detections are obtained : 

```
python3 launch_extractorDetection.py
```

The algorithm is the same of "Robust People Detection Using Depth Information" this article is located in ./Bibliography/ repository.

It follows the next steps :

* An image is cuted in regions
* Then highest heights are obtained for each region and stored in a matrix
* Candidates are obtained with which are the heighest heights in a certain neighborhood
* Candidates that are in the same neightborhood are consider as a single one
* Regions of body are calculated

Data extraction are store in : ./training/detections/v{*version*}/
For each version a video and a csv file are stored with detections and a video with the bounding boxes.

##Data Features

Before launch data festures: 
* First a global id might be defined for each person
* Then this id global is added in the csv dectection files

To extract features : 

```
python3 launch_dataExtractor.py
```

A file with the features is stored in : ./data/features.csv




## Output stream 
![alt text](pictures/stream_output.png)


## MSD
![alt text](pictures/MSD.png)
