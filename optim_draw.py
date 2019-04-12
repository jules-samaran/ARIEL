
from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
from model_encoder import *


class Drawer:
    def __init__(self, input_img, imsize):
        self.drawing = torch.zeros([imsize, imsize, 3], dtype=torch.int32)
        self.input_img = input_img
        # line_drawer is a simple MLP who draws a line on the drawing
        self.line_drawer = LineDrawer()

    def run_segment_optimizer(self, model, n_epochs=50):
        print('Optimizing the drawing..')
        epoch = 0
        # reinitializing optimizer at each segment or not?
        while epoch <= n_epochs:
            loss = model.loss(self.input_img, self.line_drawer(self.drawing))
            loss.backward()


class LineDrawer:

    def __init__(self):
        self.start_point = torch.tensor([0, 0])
        self.end_point = torch.tensor([0, 0])
        self.width = torch.tensor(5)

    def forward(self, current_drawing):
        # adding a line with tensor operations (Kubik)
        pass


# main function
def run():
    model = CNNFeatureExtractor()

    # retrain the model on small datasets containing hand drawn sketches
    model.fine_tune_model(["url_sketches"])
