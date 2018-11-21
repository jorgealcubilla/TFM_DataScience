# Highly Congested Scenes Analysis (i.e.: Traffic)

##

## Introduction:
The problem of counting number of object instances visible in a single image becomes even more chellenging in highly dense scenarios which include additional complexity due to factors like perspective, occlusion, clutter, and few pixels per object. <br>
In extremely dense populations, counting by human detection in such images is almost impossible.

## Project Phases: 

This project will address this problem by going through 4 phases: 

### 1) Selection and analysis of a deep learning model using Keras
The selected model is based on the paper ['Towards perspective-free object counting with deep learning' by Daniel Oñoro-Rubio and Roberto J. López-Sastre.](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf)

The main challenge of this section will be to translate the Caffe´s original model into Keras´.
The detailed process can be found in the notebook [i_model_selection](https://github.com/jorgealcubilla/TFM_DataScience/blob/master/i_model_selection.ipynb) within the 'master' folder of this project.

### 2) Data_set: Selection and validation
This project will use the 'TRaffic ANd COngestionS (TRANCOS)' dataset which is used by the aforementioned paper in its section# 4.1.

TRANCOS is a dataset designed for vehicle counting in traffic jam scenes, so, it is a perfect fit for this project.

For further information and instructions on how to download this dataset properly, see notebook [ii_TRANCOS_db](https://github.com/jorgealcubilla/TFM_DataScience/blob/master/ii_TRANCOS_db.ipynb) within the 'master' folder of this project.

### 3) Selected model test
In this section, we will replicate test results on TRANCOS dataset obtained in the paper mentioned above.

We will use the Keras model built in section# 1 and load parameters trained with Caffe.<br>
This is a hard job that we can complete thanks to the Github project ['caffe_weight_converter'](https://github.com/pierluigiferrari/caffe_weight_converter) by Pierluigi Ferrari.

The whole process is in notebook [iii_model_test](https://github.com/jorgealcubilla/TFM_DataScience/blob/master/iii_model_test.ipynb) of this project. 

### 4) Analysis for improvement
We will analize factors associated with the effectiveness of the model and will look at ways to improve it.
See notebook [iv_analysis_for_improvement]
(https://github.com/jorgealcubilla/TFM_DataScience/blob/master/iv_analysis_for_improvement.ipynb) of this project.

### 5) Practical application
Accurately estimating objects from images or videos has become an increasingly important application of computer vision technology for purposes of crowd control and public safety. 

In some scenarios, such as public rallies and sports events, the number or density of participating people is an essential piece of information for future event planning and space design. 

Good methods of crowd counting can also be extended to other domains, for instance, counting cells or bacteria from microscopic images, animal crowd estimates in wildlife sanctuaries, or estimating the number of vehicles at transportation hubs or trafﬁc jams, etc 

Since the model analyzed and tested in this project has been specifically trained with images from traffic cameras, I have selected an application related to traffic density estimation as an example of practical application of this project.

#### Example:
The selected example is based on a Python script called “traffic_script.py” (file included in this project) that collects an image from a traffic webcam and saves:
- The estimated traffic density of the last image on a ".txt" file (including date_time of extraction).<br>
This file could be used to provide information to a webpage (see below). <br>
- The image and its estimated traffic density in a **history dataset**, which will allow further improvement of the model (i.e.: analysis, training, test, …) and deep analysis of traffic data (i.e.: traffic density evolution, traffic density prediction for the rest of the day/week, …).

A specific application of this script can be found on:<br>
https://jorgealcubilla.github.io/traffic_density/ , <br>
from: https://github.com/jorgealcubilla/traffic_density

In addition to user instructions included in the script, the following information is also relevant for this specific application:

Traffic Camera selection: Although the traffic cameras used to train and test our model are available, they constantly change their focusing what means that the extension of the area subject to analysis is variable.

As we need a fixed area for comparison purposes, I selected a traffic camera from another source. <br>
Traffic cameras from http://www.drivebc.ca/ have a fixed focus, so they are perfect for this example. <br>
I selected one of them.

Automation: As the selected traffic camera updates the image every 15 minutes, I have scheduled a task on my operating system (Windows) to automate the Python script execution and commit&push the prediction from my local computer to the Github repository of the website https://github.com/jorgealcubilla/traffic_density. <br>
This way, both, history dataset and website, are updated at the same time as the image from the traffic camera. <br>

The task is executed only during the daytime.
 
## Conclusions:
This project is an approach to machine learning applied to congested scenes using a methodology based on density maps.

It has overcome difficulties such as:
- Find a model that is documented enough to be implemented properly.
- Make thorough analysis/validation of the dataset, detecting minor errors and room for improvement.
- Replicate a Caffe model into Keras
- Translate model parameters from Caffe into Keras

Then, a personal methodology for testing and validation purposes has been implemented and carried out a deep analysis of the selected model.

Some contributions of this project:

- Making machine learning technics based on density maps, and the selected model, much more understandable and easier to implement.
- Providing a practical example of: parameters collection, from Caffe models, for advanced Keras models. 
- Providing a strategy for improvement of the selected model

As a result, the selected model was easily implemented in a Python script that provides estimations of traffic level in real time, which is a valuable source of data for further applications such as traffic analysis and forecast.  




