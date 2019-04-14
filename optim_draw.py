
from __future__ import print_function
import torch.optim as optim
import copy
from model_encoder import *


class Drawer:
    def __init__(self, input_img, imsize=imsize):
        self.drawing = torch.ones([1, 3, imsize, imsize], dtype=torch.float32)
        self.input_img = input_img
        # line_drawer is a simple MLP who draws a line on the drawing
        self.line_drawer = LineDrawer()

    def run_segment_optimizer(self, cnn, n_epochs=20):
        print('Optimizing the line..')
        epoch = 0
        # reinitializing optimizer at each segment or not?
        optimizer = optim.Adam([self.line_drawer.start_point.requires_grad_(),
                                self.line_drawer.end_point.requires_grad_()])
        while epoch <= n_epochs:
            print("epoch %i out of %i" % (epoch, n_epochs))
            optimizer.zero_grad()
            loss = cnn.comparison_loss(cnn.model(self.line_drawer.forward(self.drawing)))
            loss.backward()
            optimizer.step()
            epoch += 1
            self.drawing = self.line_drawer.forward(self.drawing)


class LineDrawer:

    def __init__(self):
        self.start_point = torch.tensor([0, 0], dtype=torch.float32)
        self.end_point = torch.tensor([100, 100], dtype=torch.float32)
        self.width = 5

    def forward(self, current_drawing):
        length = torch.dist(self.start_point, self.end_point)

        # determinant for line width
        det = torch.tensor([[(j-self.start_point[0]) * (self.end_point[1] - self.start_point[1]) -
                             (i-self.start_point[1]) * (self.end_point[0] - self.start_point[0])
                             for j in range(imsize)] for i in range(imsize)], dtype=torch.float32, requires_grad=True)

        # scalar product to test belonging to the segment
        scal = torch.tensor([[(j - self.start_point[0]) * (self.end_point[0] - self.start_point[0]) +
                             (i - self.start_point[1]) * (self.end_point[1] - self.start_point[1])
                             for j in range(imsize)] for i in range(imsize)], dtype=torch.float32, requires_grad=True)

        # combining the above tensors to obtain the line, using sigmoids for differentiability
        line = torch.ones([imsize, imsize]) -\
            torch.sigmoid(10*(det+self.width)) * torch.sigmoid(10*(self.width-det)) \
            * torch.sigmoid(10*scal) * torch.sigmoid(10*(length*length-scal))

        # putting the line in the format [1, 3, imsize, imsize]
        line13 = line.unsqueeze(0).expand(3, imsize, imsize).unsqueeze(0)

        # returning a copy of the drawing with the line
        drawing_copy = torch.tensor(current_drawing, requires_grad=True)
        return drawing_copy*line13


# main function
def run(n_lines):
    cnn = CNNFeatureExtractor()

    # retrain the model on small datasets containing hand drawn sketches NOT YET
    # model.fine_tune_model(["url_sketches"])
    for param in cnn.model.parameters():  # the cnn feature extractor has already been trained, we freeze its parameters
        param.requires_grad = False
    input_img = image_loader("./Images/boat.jpg")
    cnn.add_comparison_loss(input_img)
    drawer = Drawer(input_img)
    for k in range(n_lines):
        print("Drawing line number %i" % k)
        drawer.run_segment_optimizer(cnn)
        imshow(drawer.drawing)
