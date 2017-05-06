**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.mdsummarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy
####1. Model Architecture
The model is from line 54 to line 75 in `model.py`.

The first step is to normalize the images.

The scond step is to crop the image. Here I cut the top part which is usually sky and is useless for us to train.

Then it is the neural networks.There are Convolutional layers and each one has a MaxPooling and Dropout layers following the convoluiton.

Then flatten the output of convolution layers. 4 layers of fully connected layers are added, and each uses the `relu` as activation function. The output is like a regression value, which is a steering angle. 

Adam optimizer is used here to tune the parameters. Since the training set is very large in this project, generator is used here to save the memory.

And the model is saved in `model.h5`.

####2. Creation of the Training Set & Training Process
TO augment the data, all images (left, right, center) are included here, and 0.5 degree is as offset to added in the steering angle as output.
To augment the data set, I also flipped images and angles thinking that this would give more training data. 

Also, after first training, when I tested the trained model, I found the model performed badly in the left corner after bridge since there is no right lane mark. Then I collected the data in this area several times.


After the collection process, I had 10248 frames and each frame has three images.I finally randomly shuffled the data set and put 20% of the data into a validation set. Therefore I trained 8198 frames. Since I flipped the image, the total number of training set is 8198*6, which is 49188. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by valiation loss cannot reduce any more after thrid epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
