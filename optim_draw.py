
from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
from model_encoder import *


class Drawer:
    def __init__(self, input_img, imsize=imsize):
        self.drawing = torch.zeros([1, 3, imsize, imsize], dtype=torch.int32)
        self.input_img = input_img
        # line_drawer is a simple MLP who draws a line on the drawing
        self.line_drawer = LineDrawer()

    def run_segment_optimizer(self, model, n_epochs=50):
        print('Optimizing the drawing..')
        epoch = 0
        # reinitializing optimizer at each segment or not?
        optimizer = optim.Adam([self.line_drawer.start_point.requires_grad_(),
                                self.line_drawer.end_point.requires_grad_()])
        while epoch <= n_epochs:
            optimizer.zero_grad()
            loss = model.loss(self.input_img, self.line_drawer.forward((self.drawing)))
            loss.backward()
            optimizer.step()
            self.drawing = self.line_drawer.forward(self.drawing)


class LineDrawer:

    def __init__(self):
        self.start_point = torch.tensor([0, 0])
        self.end_point = torch.tensor([100, 100])
        self.width = 5

    def forward(self, current_drawing):
        drawing_with_line=copy.deepcopy(current_drawing)
        # Creating a boolean tensor with value true inside the line between start_point and end_point
        line=torch.tensor([[(j-self.start_point[0])*(self.end_point[1]-self.start_point[1])-(i-self.start_point[1])*(self.end_point[0]-self.start_point[0])<self.width for j in range(imsize)] for i in range(imsize)])


# main function
def run(n_lines):
    cnn = CNNFeatureExtractor()

    # retrain the model on small datasets containing hand drawn sketches NOT YET
    # model.fine_tune_model(["url_sketches"])
    for param in cnn.model.parameters():  # the cnn feature extractor has already been trained, we freeze its parameters
        param.requires_grad = False
    input_img = image_loader("./Images/dancer.jpg")
    cnn.add_comparison_loss(input_img)
    drawer = Drawer(input_img)
    for k in range(n_lines):
        drawer.run_segment_optimizer(cnn.model)
        imshow(drawer.drawing)
