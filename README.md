# Dice Detection for Board Games

### Work in progress
<br/><br/>
I'm developing an object detection model to predict bounding boxes for dice counting for board games with a colleague, using pytorch and YOLOv5. 
<br/><br/>
## Problem statement
A colleague of mine loves tabletop board games. He frequently attends gatherings of board game players and plays games with sophisticated rules and settings. A common component of the games is the reliance on 6-sided cube dice rolls to produce random results, determining the outcome of many in-game actions, such as how many steps a piece moves, whether an attack hits or misses, or how much treasure one finds upon opening a treasure chest etc. <br/><br/>
Therefore, to maintain fairness, it is crucial to ensure the dice are unbiased and the probability of each side facing up is equal. In reality, players tend to bring their own set of dice to a gaming session. And because it tends to be time consuming and tedious to statistically track the outcome of dice rolls, there is a lack of viable method of checking for use of loaded dice, which have been tampered with to favour landing on a certain side over others. <br/><br/>
This is where a Deep Learning-based computer vision solution comes in, because this is an ideal case for automation. I will develop an end-to-end pipeline that takes in images of rolled dice of various types on a variety of backgrounds (tables and game boards), detect and localise the dice within the image, and classify the dice based on the side facing up. The predictions will be output to a log file that compiles the results for one set of dice over the entire game session, which, at the end of the session, produces statistics for determining the fairness of the dice. <br/><br/>
## Plan of action
1) I have found a public [dataset](https://public.roboflow.com/object-detection/dice/) of (approximately) isometric images of dice with class labels and bounding boxes, some on a table and some on a Catan board. An example is shown below. I will use this dataset for pretraining a yolov5 model in PyTorch;***<--Currently Here***
2) I will collect a relatively small set of data using my colleague's own set of dice, on a background and under lighting conditions comparable to in a *real* gaming session; 
3) I will label the images by drawing bounding boxes around the dice in the images, specifically around the top side only, because in the isometric view sides of the dice facing sideways are also visible and can potentially confuse the model;
4) I will fine tune the pretrained model on this new dataset;
5) If the performance is good enough (>0.95 on the F1 score metric on each individual class as tentatively agreed), we will deploy this model to run on a PC which my colleague will bring to the next gaming session and test it *in production*;
6) If the performance degrades on *real* data, I will add images collected there to the training data and tune the model again.

<br/><br/>
![dice image](https://i.imgur.com/ItN4AEk.png)<br/>
Example from the public dice dataset<br/>
<br/>
## Change log

v01.2 Constructed data augmentation pipeline with Albumentations, which also transforms bounding boxes and labels accordingly<br/>
v01.1 Constructed pytorch datasets and bounding box visualisation<br/>
### 09.08.2021
v01 Constructed dataframe for metadata<br/>
