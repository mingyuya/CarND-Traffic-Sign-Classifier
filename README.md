# **Write-up for Traffic Sign Recognition** 

## Name : Matt Kim

[//]: # (Image References)

[image1]: ./figures/data_summary.png "1st_graph"
[image2]: ./figures/augmented_data_summary_.png "2nd_graph"
[image3]: ./figures/grayscale.png "Grayscale"
[image4]: ./new_images/1_big.jpg "Traffic Sign 1"
[image5]: ./new_images/2_big.jpg "Traffic Sign 2"
[image6]: ./new_images/3_big.jpg "Traffic Sign 3"
[image7]: ./new_images/4_big.jpg "Traffic Sign 4"
[image8]: ./new_images/5_big.jpg "Traffic Sign 5"

## Data Set Summary & Exploration

By using one attribute(.shape) and one numpy function(np.unique), it was possible to find the following values. (In the 2nd cell)

* The size of training set is **12630**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

The following is the visualization of the number of training examples which is correspond to each class.

![alt text][image1]

(In addition, I plotted the part of the training examples in jupyter notebook.)

## Design and Test a Model Architecture
### Pre-processing

#### Step 1) Balancing the number of training examples & Gamma variation

At first, I didn't consider to adapt step1 but, the model I trained confused one of the new images with another similar sign. After that, I realized that the model was overfitted to the signs having many examples. Therefore, I duplicate the examples which are belong to the class having small amount of examples and append it to given training set. Gamma variation of the duplicated images is done concurrently to make them an effect by using **exposure from skimage**.

The following is the number of training examples after step1.

![alt text][image2]

#### Step 2) Grayscale

There are images with same sign and different chroma - Some images are bluish and some other image are reddish, although those belong to a same class. Therefore I thought that the RGB value may not be useful data and decided to remove colors from the examples to make the model concentrate to a shape of sign. Grayscaled images were made by taking an average of RGB channels. (See the cell : Converting to Grayscale)

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

#### Step 3) Shuffle

I shuffled the training set to make the examples are randomly selected at every epoch.



### Model Architecture

#### Layers of my model

The code for my final model - LeNet_mofified - is located in the cell below the title of "Model Architecture".

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Convolution 5x5	      	| 1x1 stride, valid padding, outputs 24x24x24 				|
| RELU					|												|
| Max pooling 2x2 | 2x2 stride, valid padding, outputs 12x12x24 |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x32      									|
| Flatten |  |
| Fully connected		| 3200 fan-in, 1600 fan-out        									|
| RELU					|												|
| Dropout | keep probability : 0.7 |
| Fully connected		| 1600 fan-in, 800 fan-out        									|
| RELU					|												|
| Dropout | keep probability : 0.8 |
| Fully connected		| 800 fan-in, 200 fan-out        									|
| RELU					|												|
| Dropout | keep probability : 0.9 |
| Fully connected		| 200 fan-in, 100 fan-out        									|
| RELU					|												|
| Fully connected		| 100 fan-in, 43 fan-out        									|
| Softmax				|         									| 

**I started to build the model from the LeNet** given in Lesson 8, so I didn't replace AdamOptizer which is known as one of the well-performing optimizer. I intended to make batch size and epoch be large enough and fixed to 100 and 250. On the other hand, learning rate was chosen experimentally. There was large fluctuation of valid accuracy while training is proceed when I choose 0.001 as the rate. Finally, the value of 0.0005 showed the best result. 

The training was processed through 4 cells below the title of "Train, Validate and Test the Model". The accuracy of the model was estimated in the cell right after those cells.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

#### My final model results were:
* Validation set accuracy of **~93.0%**
* Test set accuracy of **100.0%** 

To find the solution,

I started from the well known architecture - LeNet - as I mentioned above. But, I thought that I need more and deeper convolutional layer to perceive more complex feature compared to MNIST dataset. Therefore I add one more convolutional layer and increased the depth of filter. As a result, I got huge amount of parameters at fully connected layers. That is the reason why I had to place some dropout layers after the activations to prevent overfitting. However, that was not enough to get the accuracy higher than 90%. I found that the value of 0.04 that is smaller than initial value (=0.1) showed improvement.
 
###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and the last images might be difficult to classify. Because the first image is belongs to one of the classes having lack of example and has distortion by being taken at lower position. On the other hand, the last image has some darkness and uncovered part on the sign by snow.

On the contrary to the expectation above, the model predicted the classes of the new images exactly - It means that the accuracy is 100%. It is shown in the cells below the title of "Predict the Sign Type of Each Image".

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work    									| 
| General caution     			| General caution 										|
| Dangerous curve to the right					| Dangerous curve to the right											|
| Speed limit, 30 km/h	      		| Speed limit, 30 km/h					 				|
| Beware of ice/snow			| Beware of ice/snow      							|

But, it is possible to check that there the model is possible to confuse the last image as 'Children crossing' sign. The other images were clearly identified relative to the last image. The result is in the cells below the title of "Output Top 5 Softmax ..."
