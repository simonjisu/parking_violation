# Parking Violation Project

## TODO

* [ ] Post Processing of segmented(wipe out the terrain and sky)
* [ ] identifing the road lane by color 
* [ ] segmented mask of road to see which part of road will be good for parking
* [ ] construct the parking available map with gps
* [ ] detecting the car & license on a road
* [ ] building a system to judge whether the car is violating or not

## Working Log

### 2020.10.13 

* Starting the project, tried to explain the background of "parking violation", why i want to do it and how to do it

> * Motivation: Parking Violation cause some social problems
>     - Traffic jams on the roads and alleys
>     - Obstructing the passage of fire trucks or ambulances (Jecheon Sports Center Fire Accident)
>     - Blind the driver and cause the pedestrians accident, especially in children protection area
> * Merit
>     - Reduction of the traffic jams and accidents
>     - the fire trucks and ambulances are not stacked by the cars
> * Proposed Solution
>     - Using single surveillance camera to detect parking violation

* Presentation [Link](https://docs.google.com/presentation/d/1Af0SZjW4oYmFpdPG1FyUAO3Z2HnYC5EpY74rYYtB3UI/edit?usp=sharing) 

### 2020.10.28

* Simplify the complicate problem into detecting only whether the car can be parked or not by line color. 
* 2 Step Scenario:
    1. construct the map: detect the lane color using normal camera and record the data in every blocks with gps coordinate
    2. Detection(car & license) with snapshot using front camera.
* 1 Step Scenario: 
    - Detection with front camera, depth camera 

* Presentation [Link](https://docs.google.com/presentation/d/1Ahgvk7S3Tn3T3DzfhwayoOMTe-9A2Qn5V91xyOyq6wI/edit?usp=sharing)

### 2020.11.18

* Try to using Jetson Nano with RealSense Camera to record the sample video.
* Found out that don't really need to record the side of road to detect lane colors, recording the front of the road can also do the same thing.

### 2020.12.09

* Tyr to detect the lane of a road with traditional computer vision techniques
* Details are in github notebook: [02_LaneDetection_Adv.ipynb](https://nbviewer.jupyter.org/github/simonjisu/parking_violation/blob/master/notebooks/02_LaneDetection_Adv.ipynb)
* Video path: `notebooks/videos/video_output.mp4`

### 2020.12.31

* Recorded a sample video to do semantic segmentation. 
* The reason to do it is have to know which part of the video is a road or a pedestrain road or other things. So that we can combine the lane image mask to determine if this area can park or not
* Since the bottom of video contains the front of car, have to cut bottom part of image to do more acurrate segmentation. This cause some noise of classify some pixel to the car label.
* The output still has some noise when classify the image, I am going to apply CRF technique to do post-processing to get the segmented road
* Video path: `notebooks/videos/sample1_480x640_resnest_result.mp4`

