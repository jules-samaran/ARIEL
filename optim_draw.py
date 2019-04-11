
from __future__ import print_function
import torch
import torch.optim as optim
from model_encoder import *


class Drawer:
    def __init__(self, input_img, imsize):
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.drawing = None
        self.input_img = input_img
        # line_drawer is a simple MLP who draws a line on the drawing
        self.line_drawer = LineDrawer()

    # ajoute un segment sur l'image tensorielle aux coordonnées normalisées
    def draw_line_tensor(self, x1, y1, x2, y2):
        # do tensor only operations to draw a line on the image tensor
        pass

    def run_segment_optimizer(self, model, n_epochs=50):
        print('Optimizing the drawing..')
        epoch = 0
        # reinitializing optimizer at each segment or not?
        while epoch <= n_epochs:
            loss = model.loss(self.input_img, self.line_drawer())
            loss.backward()


class LineDrawer:
    def __init__(self):
        pass

    def forward(self):
        pass


def get_input_optimizer(point_1, point_2):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([point_1.requires_grad_(), point_2.requires_grad_()])
    return optimizer


# main function
def run():
    model = CNNFeatureExtractor()

    # retrain the model on small datasets containing hand drawn sketches
    model.fine_tune_model(["url_sketches"])
