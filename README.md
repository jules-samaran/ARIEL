# ARIEL : Automated Reconstruction of Images with Enhanced Lines

This project was conducted on their free time by two students from Mines Paristech, Jules Samaran and Timothée Launay. The objective was to create an algorithm which takes as input an image and draws a sketch as close as possible to the input image. We chose to restrict the drawing technique to black straight lines drawn iteratively so that the algorithm's behaviour would resemble an artist's attempt to draw a preliminary sketch. In the "Technical description" section you can read how we achieved this goal.

## Getting started

### Installing

Start by cloning this repository and in the directory where you downloaded the project run this command in your Linux terminal:
```
pip install -r requirements.txt
```
or simply install manually every python package listed in the requirements.txt file.

### Using ARIEL

Take a look at "Demo_notebook.ipynb" which will show what the algorithm can do and will enable you to realize sketches of any image.

## Technical description

We started with the idea that we wanted to draw each line one by one by adding at each iteration the line that minimizes the comparison loss between the unfinished drawing and the input image.

### The content loss

Thus we needed to come up with a relevant criteria able to quantify the difference between the CONTENT of two images.

#### Convolutional Neural Networks

Convolutional Neural Networks (CNNS) are the class of deep neural networks s that are most powerful in image processing tasks. They can be described as successive layers of collections of image filters that each extract different features from the input image. When Convolutional Neural Networks are trained on object recognition, they develop a representation of the image that makes object information increasingly explicit along the processing hierarchy. We chose to use [this pre-trained model](https://arxiv.org/abs/1409.1556) because we could not hope to train on our own a model as performant as this one. This model was initially trained on a huge dataset for image classification, the output of the feature maps would be fed to fully connected layers whose output would determine the class to which the input image belonged to. We downloaded this whole model and troncated by removing all the layers after the one feature map we were interested in and we froze all the parameters in the model.

The content loss we used then was the mean squared difference between the encoded versions of both images. What we call the encoded version of an image is the output of the forward computation of one feature map by the CNN. We chose a feature map commonly used to characterize the content of an image. 

### Initializing and optimizing each drawn line

The next step after devising this content loss was to be able to automatize the solving of a minimization problem: at each iteration when we draw a new line on the current drawing, which line minimizes the content loss between the drawing and the target image.

What we optimized were 4 parameters only: the coordinates of the two extremities of the line being drawn. We had three other parameters which were stationary: width (the width of the line), intensity (a float number between 0 and 1 corrsponding to the intensity of the pixels on the line) and decay (float that determines how fast the pixels become white arounf the line in terms of number of pixels).

CNNs treat the information locally, the problem is that if we only deal with extremely thin straight lines, if the line we are drawing is not initially extremely close to the target line we are trying to draw then we won't be able to optimize the parameters of the drawn line. Indeed outside of a very small sphere of pixels around the reference line, being 10 pixels or 60 pixels away from the target line doesn't make any difference from the point of view of the CNN's comparison loss. Since the gradient of the comparison loss will be zero, we won't be able to move step by step the drawn line closer to the target line by using a gradient descent.

#### Selection among random initializations

In order to give the algorithm a chance to solve this minimization problem we started each iteration by choosing randomly 100 or 200 initializations for the drawn line and selecting the one for which the loss is the lowest when we add it to the drawing.

#### The blurrying trick

We decided to draw wider lines and to raise significantly the decay parameter so that each drawn line actually affects several pixels around it and is less restrictively localized.
This trick allowed the optimization to be completed without any problems, you can see more about this in the first section of the notebook.


## Contributing

If you have any question or suggestion, raise an issue on this github repository and we'll get back to you promptly.

## Authors and acknowledgements

Timothée Launay and Jules Samaran, with special thanks to Mathieu Kubik and Aznag Abdellah for helpful discussions.
