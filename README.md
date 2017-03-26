# **Behavioral Cloning**

[![Demo video](https://img.youtube.com/vi/UZ4zafFSHQc/0.jpg)](https://www.youtube.com/watch?v=UZ4zafFSHQc)

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./imgs/model.png "model"
[recover01]: ./imgs/recover01.jpg "recover01"
[recover02]: ./imgs/recover02.jpg "recover02"
[recover03]: ./imgs/recover03.jpg "recover03"
[good]: ./imgs/good.jpg "good"
[figure_1]: ./imgs/figure_1.png "figure_1"


###### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists 12 blocks, 5 convolution neural network Blocks, 2 preprocesse blocks and 5 fully connected blocks. (model.py lines 109-159)

Some convolution neural network Blocks include RELU activation layers, Max Pooling layers and dropout layers.
Preprocesse blocks include normalizing inputs and croping.
Fully connected blocks have RELU layers to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting .
```python
    model.add(Dropout(0.25))
```
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 163).
```python
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch= len(train_samples*3),
                                     validation_data=validation_generator,
                                     callbacks = [tfBoard],
                                     nb_val_samples=len(validation_samples), nb_epoch=10)
```
#### 4. Appropriate training data

The training data was choosen to keep the car driving center of teh road, make good data be the dominant data in the training sets. I also add recover data when the venhicle was off track.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for model architecture was similar to the paper [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) with some modification. The reason for using that architecture is the problem similar, therefore that should be a greate starting point for this project.

The data splited into training set and validation set, whcih can check the overfitting. The car drive out of the track in first iteration, which i collect 4 driving datasets. The reason i figured out is the problem in the datasets, which include those dangerous actions.

The car drive well in second iteration with better quality of data. For collecting the recovering data, it should avoid include those driving out of the track, the reason is avoid include those inappropriate actions into training sets. I start the recording when the car was out of the track, which can include those recover actions into model.

I added one more dataset when i try to accomplish a checkpoint, this incremental can identify which dataset have problem and recollect it. The problematic dataset can be the action going off the track and those dangrous actions, in order to avoid these actions include in the final model, these datasets should not include in the training sets.   

I use adam optimizer for training, whcih learning rate is not needed.

#### 2. Final Model Architecture

The final model architecture (model.py lines 110-164) consisted of a convolution neural network with the following layers and layer sizes.
all conv. layers use valid border mode.

| layer        | size / description | output  |
| ------------- |:-------------:| -----:|
| normalize_1 | normalize input | none, 160, 320, 3 |
| Cropping2D_1 | Crop input | None, 70, 320, 3 |
| Convolution2D | conv input 24x5x5 | none, 66, 316, 24 |
| relu | Activation | none, 66, 316, 24 |
| Dropout | Dropout | none, 66, 316, 24 |
| Convolution2D | conv input 36x5x5 | None, 62, 312, 36 |
| relu | Activation | none, 66, 316, 24 |
| MaxPooling | MaxPooling 3x3 | none, 20, 104, 36 |
| Dropout | Dropout | none, 66, 316, 24 |
| Convolution2D | conv input 36x5x5 | None, 16, 100, 36) |
| relu | Activation | none, 16, 100, 36 |
| MaxPooling | MaxPooling 3x3 | none, 8, 50, 36 |
| Dropout | Dropout |  none, 8, 50, 36 |
| Convolution2D | conv input 48x3x3 | None, 6, 48, 48 |
| relu | Activation | none, 16, 100, 36 |
| Convolution2D | conv input 64x3x3 | None, 4, 46, 64 |
| relu | Activation | none, 16, 100, 36 |
| Flatten | Flatten | none, 11776 |
| FC | FC | none, 1164 |
| relu | Activation | none, 1164 |
| Dropout | Dropout |  none, 1164 |
| FC | FC | none, 100 |
| relu | Activation | none, 100 |
| Dropout | Dropout |  none, 100 |
| FC | FC | none, 50 |
| relu | Activation | none, 50 |
| Dropout | Dropout |  none, 50 |
| FC | FC | none, 10 |
| FC | FC | none, 1 |

Here is a visualization of the architecture in tensorboard.
![model][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![good][good]

I recorded recovering video from off track to on track. (not include on track to off track)
![recover01][recover01]
![recover02][recover02]
![recover03][recover03]

After the collection process, I had 8928 number of data points. The image that use in this training is 28568. (include fliping and sides cameras)

###### Shuffle data
I randomly shuffled the data set and put 20% of the data into a validation set.
```python
sklearn.utils.shuffle(train_samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

###### left and right cameras
I also make use of left camera of right camera, which adding a correction factor to the value, it makes the duration for data collection reduced. In my testing correction 0.13 is the best.

###### flip ceneter data sets
One of the challenge is small data will makes some actions be the dominants, it's because the dataset most include those turn, for instance turning right are the most captured in clockwise mode, which makes the data sets inblance. Fliping the central camera can create a blance distribution in the datasets. In my experience it works great and improve a lot mirror center camera.

- why do not mirror left and right cameras?
    - Since left right cameras are adding a correction value, which a parameter hard to tune, it create some noise to correct actions. Therefore it makes the car keep turning even in straight road.      

![figure_1][figure_1]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by MSE reduce much more slowly. I used an adam optimizer so that manually training the learning rate wasn't necessary.
