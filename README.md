# ARIEL

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

Thus we needed to come up with a relevant criteria able to quantify the difference between the CONTENT of two images. blabla on CNNs. We chose to use [this pre-trained model](https://arxiv.org/abs/1409.1556) because we could not hope to train on our own a model as performant as this one.

### Initializing and optimizing each drawn line

#### Selection among random initializations

#### The blurrying trick

## Contributing

If you have any question or suggestion, raise an issue on this github repository and we'll get back to you promptly.

## Authors and acknowledgements

Timothée Launay and Jules Samaran from Mines Paristech, with special thanks to Mathieu Kubik and Aznag Abdellah for helpful discussions.
