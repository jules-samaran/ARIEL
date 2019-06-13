
from __future__ import print_function
import torch.optim as optim
from model_encoder import *
from tqdm import tqdm
import numpy as np
import time


class Drawer:
    # this is the object that oversees the whole drawing process
    def __init__(self, input_img, imsize=imsize):
        self.drawing = torch.ones([1, 3, imsize, imsize], dtype=torch.float32)
        self.input_img = input_img
        self.line_drawer = LineDrawer()
        self.loss_history = []  # this contains the evolution of the loss durig each optimization
        self.line_history = []  # this will contain all the coordinates of the lines drawn

    # Draws a new line
    def run_segment_optimizer(self, cnn, n_epochs=10):
        print('Initializing the line..')
        self.line_drawer = LineDrawer()
        min_loss = cnn.comparison_loss(cnn.model(self.line_drawer.forward(self.drawing)))
        for i in range(200):
            test_line = LineDrawer()
            test_drawing = test_line.forward(self.drawing)
            test_loss = cnn.comparison_loss(cnn.model(test_drawing))
            if test_loss < min_loss:
                self.line_drawer = test_line
                min_loss = test_loss
                print("Current best line initialization loss: ", min_loss.item())
        # Display only the best initialization
        imshow(self.line_drawer.forward(self.drawing))

        print('Optimizing the line..')
        optimizer = optim.Adamax([self.line_drawer.start_point.requires_grad_(),
                                  self.line_drawer.end_point.requires_grad_()], lr=5)

        for epoch in tqdm((range(1, n_epochs + 1))):

            def closure():
                optimizer.zero_grad()
                loss = cnn.comparison_loss(cnn.model(self.line_drawer.forward(self.drawing)))
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            self.loss_history.append(loss.item())

        print("Final loss : ", loss.item())
        self.drawing = self.line_drawer.forward(self.drawing)
        imshow(self.drawing)

        self.line_history.append([self.line_drawer.start_point.clone(),self.line_drawer.end_point.clone()])


class LineDrawer:
    # this object just handles how we add a line on a drawing
    def __init__(self):
        self.start_point = imsize * torch.rand([2])
        self.end_point = imsize * torch.rand([2])
        self.width = torch.tensor(1.0 * imsize/64, dtype=torch.float32)
        self.decay = torch.tensor(2.0/(imsize/64), dtype=torch.float32)
        self.intensity = torch.tensor(0.3, dtype=torch.float32)

    def forward(self, current_drawing):
        length = torch.dist(self.start_point, self.end_point)

        i_values = torch.tensor([[i for j in range(imsize)] for i in range(imsize)], dtype=torch.float32)
        j_values = torch.tensor([[j for j in range(imsize)] for i in range(imsize)], dtype=torch.float32)

        start_point_x = self.start_point[0].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize,imsize)
        start_point_y = self.start_point[1].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize,imsize)
        end_point_x = self.end_point[0].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize,imsize)
        end_point_y = self.end_point[1].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize,imsize)

        # determinant for line width
        det = (j_values-start_point_x) * (end_point_y - start_point_y) -\
              (i_values-start_point_y) * (end_point_x - start_point_x)

        # scalar product to test belonging to the segment
        scal = (j_values - start_point_x) * (end_point_x - start_point_x) +\
               (i_values - start_point_y) * (end_point_y - start_point_y)

        # combining the above tensors to obtain the line, using sigmoids for differentiability
        line = torch.ones([imsize, imsize]) -\
            torch.sigmoid(self.decay*(det/length+self.width/2)) * torch.sigmoid(self.decay*(self.width/2-det/length)) \
            * torch.sigmoid(self.decay*scal/length) * torch.sigmoid(self.decay*(length-scal/length)) \
            * self.intensity

        # putting the line in the format [1, 3, imsize, imsize]
        line13 = line.unsqueeze(0).expand(3, imsize, imsize).unsqueeze(0)

        # returning a copy of the drawing with the line
        drawing_copy = current_drawing.clone().detach()
        output = drawing_copy*line13
        return output


# main function
def run(input_img, n_lines, n_epoch=40, unblur=True, save=True, save_title='untitled', save_points=False):
    r""" This is the function we use to ask ARIEL to draw a sketch of an image

    :param input_img: the image we want to draw
    :param n_lines: an integer that corresponds the number of straight lines used to draw the sketch, it is determined
    before we start drawing
    :param n_epoch: an integer, it is the number of iterations in the optimization of each line
    :param unblur: a boolean that determines whether or not we will show at the end of the computations another version
    of the drawing where the lines will have been unblurred
    :param save: a boolean that indicates whether you want to save the final drawing
    :param save_title: a str that indicates the path here you would like to save the results if save were True
    (you don't need to add the .jpg extension to that path)
    :param save_points: a boolean that indicates whether you want to save the coordinates of the extremities of the
    lines drawn
    :return: the drawer object
    """

    t_0 = time.time()
    cnn = CNNFeatureExtractor()

    # retrain the model on small datasets containing hand drawn sketches NOT YET
    # model.fine_tune_model(["url_sketches"])
    for param in cnn.model.parameters():  # the cnn feature extractor has already been trained, we freeze its parameters
        param.requires_grad = False
    cnn.add_comparison_loss(input_img)
    drawer = Drawer(input_img)

    # Drawing the lines
    for k in range(n_lines):
        print("Drawing line number %i" % k)
        drawer.run_segment_optimizer(cnn, n_epoch)

    # Saving results
    image_title=save_title + '_drawing.jpg'
    imshow(drawer.drawing, title=image_title, save=save)

    if save_points:
        points_title = save_title + '_segment_coordinates'
        np.save(points_title, drawer.line_history)

    if unblur:
        unblurred_image = torch.ones([1, 3, imsize, imsize], dtype=torch.float32)
        unblurred_line_drawer = LineDrawer()
        unblurred_line_drawer.width = 1.5*(imsize/64)
        unblurred_line_drawer.decay = 6/(imsize/64)
        unblurred_line_drawer.intensity = 0.2
        for line in drawer.line_history:
            unblurred_line_drawer.start_point=line[0]
            unblurred_line_drawer.end_point=line[1]
            unblurred_image = unblurred_line_drawer.forward(unblurred_image)
        unblurred_title = save_title + '_unblurred_drawing.jpg'
        imshow(unblurred_image, title=unblurred_title, save=save)
    # computing and formatting the execution time
    t_end = time.time()
    secs = t_end - t_0
    mins = secs // 60
    runtime = [mins // 60, mins % 60, secs % 60]
    print("Complete runtime: %ih %im %is"% (runtime[0], runtime[1], runtime[2]))
    return drawer
