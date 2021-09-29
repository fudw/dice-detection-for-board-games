# Dice Detection for Board Games

### *** Work in progress ***
<br/><br/>
I'm developing an object detection model to predict bounding boxes for dice counting for board games with a colleague, using pytorch and YOLOv5. 
<br/><br/>
## Problem statement
A colleague of mine loves tabletop board games. He frequently attends gatherings of board game players and plays games with sophisticated rules and settings. A common component of the games is the reliance on 6-sided cube dice rolls to produce random results, determining the outcome of many in-game actions, such as how many steps a piece moves, whether an attack hits or misses, or how much treasure one finds upon opening a treasure chest etc. <br/><br/>
Therefore, to maintain fairness, it is crucial to ensure the dice are unbiased and the probability of each side facing up is equal. In reality, players tend to bring their own set of dice to a gaming session. And because it tends to be time consuming and tedious to statistically track the outcome of dice rolls, there is a lack of viable method of checking for use of loaded dice, which have been tampered with to favour landing on a certain side over others. <br/><br/>
This is where a Deep Learning-based computer vision solution comes in, because this is an ideal case for automation. I will develop an end-to-end pipeline that takes in images of rolled dice of various types on a variety of backgrounds (tables and game boards), detect and localise the dice within the image, and classify the dice based on the side facing up. The predictions will be output to a log file that compiles the results for one set of dice over the entire game session, which, at the end of the session, produces statistics for determining the fairness of the dice. <br/><br/>
## Plan of action
1) I have found a public [dataset](https://public.roboflow.com/object-detection/dice/) of (approximately) isometric images of dice with class labels and bounding boxes, some on a table and some on a Catan board. An example is shown below. I will use this dataset for pretraining a yolov5 model in PyTorch;
2) I will collect a relatively small set of data using my colleague's own set of dice, on a background and under lighting conditions comparable to in a *real* gaming session;
3) I will label the images by drawing bounding boxes around the dice in the images, specifically around the top side only, because in the isometric view sides of the dice facing sideways are also visible and can potentially confuse the model;  ***<--Currently Here***
4) I will fine tune the pretrained model on this new dataset;    
5) If the performance is good enough (>0.95 on the F1 score metric on each individual class as tentatively agreed), we will deploy this model to run on a PC which my colleague will bring to the next gaming session and test it *in production*;
6) If the performance degrades on *real* data, I will add images collected there to the training data and tune the model again.

<br/><br/>
![dice image](https://i.imgur.com/ItN4AEk.png)<br/>
Example from the public dice dataset<br/>
<br/>
## Change log
v1.09 The pre-trained model performs poorly on the new dataset of different-looking dice collected in lighting and background conditions dissimiliar to the pre-training dataset. 

v1.08 Trained for 500 epochs from COCO weights with SGD optimiser and up-down random flip. Train and val losses still drop slightly more slowly than v01.5; up-down random flip doesn't seem to help with val performance either, possibly because data don't have much variation since images all taken from a similar viewpoint.

v1.07 Trained for 500 epochs from COCO weights with Adam optimiser, without up-down random flip. Train and val losses still drop more slowly than v01.5; adam optimiser seems to converge less efficiently than SGD for this model.

v1.06 Trained for 500 epochs from COCO weights with Adam optimiser instead of SGD and added p=0.5 up-down random flip augmentation. Losses drop much more slowly than v01.5; at the end of 500 epochs losses are still dropping and mAP@0.5 ~0.90 and F1 0.84 at 0.409 confidence.

v1.05 Trained for 500 epochs, losses mostly plateured. On val set mAP@0.5 0.988; F1 0.97 at 0.528 confidence for all classes; no notable variation among 6 classes.<br/>
v1.04 Trained for 100 epochs, losses still dropping. mAP@0.5 ~0.45

v1.03 Prepared files in accordance with yolov5 requirements.

v1.02 Constructed data augmentation pipeline with Albumentations, which also transforms bounding boxes and labels accordingly.

v1.01 Constructed pytorch datasets and bounding box visualisation.

### 09.08.2021
v1.0 Constructed dataframe for metadata.
