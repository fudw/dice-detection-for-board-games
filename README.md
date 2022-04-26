# Dice Detection for Board Games

### *** Work in progress ***
<br/><br/>
I'm developing an object detection model to predict bounding boxes for dice counting for board games with a colleague, using pytorch and YOLOv5. 
<br/><br/>
## Problem statement
A colleague of mine loves tabletop board games. He frequently attends gatherings of board game players and plays games with sophisticated rules and settings. A common component of the games is the reliance on 6-sided cube dice rolls to produce random results, determining the outcome of many in-game actions, such as how many steps a piece moves, whether an attack hits or misses, or how much treasure one finds upon opening a treasure chest etc. <br/><br/>
Therefore, in many games it is common to cast up to 20 dice in a single roll, and then either the sum of all points is calculated or the number of, say, sixes is counted, to determine the result of the action. This process can be quite tedious and prone-to-error for humans to perform accurately, quickly, and repeatedly. <br/><br/>
This is where a Deep Learning-based computer vision solution comes in, because this is an ideal case for automation. I will develop an end-to-end pipeline that takes in images of rolled dice of various types on a variety of backgrounds (tables, game boards, or dice trays), detect and localise the dice within the image, and classify the dice based on the side facing up. The predictions will be output to a table displaying statistics of the roll, such as the total sum or counts for each face. <br/><br/>
## Plan of action
1) I have found a public [dataset](https://public.roboflow.com/object-detection/dice/) of (approximately) isometric images of dice with class labels and bounding boxes, some on a table and some on a Catan board. An example is shown below. I will use this dataset for pretraining a yolov5 model in PyTorch;
2) I will collect a relatively small set of data using my colleague's own set of dice, on a background and under lighting conditions comparable to in a *real* gaming session;
3) I will label the images by drawing bounding boxes around the dice in the images, specifically around the top side only, because in the isometric view sides of the dice facing sideways are also visible and can potentially confuse the model;
4) I will fine tune the pretrained model on this new dataset;     
5) If the performance is good enough (>0.95 on the F1 score metric on each individual class as tentatively agreed), we will deploy this model to run on a PC which my colleague will bring to the next gaming session and test it *in production*; ***<--Currently Here*** 
6) If the performance degrades on *real* data, I will add images collected there to the training data and tune the model again.

<br/><br/>
An example from the public dice dataset used for pre-training:<br/>
![dice image](https://i.imgur.com/ItN4AEk.png)<br/>
<br/>
The results on a few validation set images after fine-tuning are shown below.
![df1fcf33-9730-44eb-b9c6-fda8daa84ecc](https://user-images.githubusercontent.com/77344869/153873206-8751345a-cdce-41ec-8a4c-794a4288e350.jpg)
<br/>
and the confusion matrix on the validation set:
![7e56739c-0688-4ca2-890d-781395814ee9](https://user-images.githubusercontent.com/77344869/153873249-e369872e-d784-4a66-9ced-24d5abfbc060.png)


## Change log
v1.12 Re-train starting from new checkpoints of yolov5 to make use of more efficient model architectures.

v1.11 As an ablation test, re-training from ImageNet weights directly on the small dataset does not produce comparable results, as the model cannot learn the right, specific features of dice faces from only the small number of examples. This confirms that the pre-training on the large dataset of dice is essential.

v1.10 Split the new dataset of 20 images into 14:6 train:val. Fine-tuned pretrained model on this small dataset. Experimented with: 1) Adam optimizer better than SGD; 2) 0.001 LR better than 0.01; 3) Allowing all layers to train better than freezing parameters of the first 10 layers; 4) 1024 input image size vastly better than 512, 2048 even better. Initial baseline performance plateaus after 500 epochs with mAP@0.5 ~0.44, F1 score 0.41 at 0.179 confidence. Best run performance plateaus after 50 epochs with mAP@0.5 ~0.995 and F1 1.0 at 0.836 confidence. The input image size heavily affects detection accuracy; inference and training should be on the same scale. 

v1.09 The pre-trained model performs poorly on the new test dataset, which consists of much higher resolution images (4032x3024) taken by an iPhone, of different-looking dice collected in lighting and background conditions dissimiliar to the pre-training dataset. Based on labels manually created by myself, for all classes Precision: 0.0836; Recall: 0.21; mAP@0.5: 0.0536; mAP@0.5-0.95: 0.0273. Changing input size, because test images are much larger than training images does not make a difference. 

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
