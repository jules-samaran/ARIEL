
from __future__ import print_function
import torch.optim as optim
from model_encoder import *


class Drawer:
    def __init__(self, input_img, imsize=imsize):
        self.drawing = torch.ones([1, 3, imsize, imsize], dtype=torch.float32)
        self.input_img = input_img
        # line_drawer is a simple MLP who draws a line on the drawing
        self.line_drawer = LineDrawer()

    def run_segment_optimizer(self, cnn, n_epochs=10):
        print('Initializing the line..')
        self.line_drawer = LineDrawer()
        min_loss = cnn.comparison_loss(cnn.model(self.line_drawer.forward(self.drawing)))
        for i in range(100):
            test_line = LineDrawer()
            test_loss = cnn.comparison_loss(cnn.model(test_line.forward(self.drawing)))
            if test_loss < min_loss:
                self.line_drawer = test_line
                min_loss = test_loss
                print(min_loss)

        print('Optimizing the line..')
        epoch = 0
        # reinitializing optimizer at each segment or not?
        optimizer = optim.Adamax([self.line_drawer.start_point.requires_grad_(),
                                  self.line_drawer.end_point.requires_grad_()], lr=5)

        while epoch <= n_epochs:
            print("epoch %i out of %i" % (epoch, n_epochs))

            def closure():
                optimizer.zero_grad()
                loss = cnn.comparison_loss(cnn.model(self.line_drawer.forward(self.drawing)))
                print(loss)
                loss.backward()
                return loss
            optimizer.step(closure)
            epoch += 1
        self.drawing = self.line_drawer.forward(self.drawing)


class LineDrawer:

    def __init__(self):
        self.start_point = imsize*torch.rand([2])
        self.end_point = imsize*torch.rand([2])
        self.width = torch.tensor(5, dtype=torch.float32)
        self.decay = torch.tensor(0.5, dtype=torch.float32)

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
            * torch.sigmoid(self.decay*scal/length) * torch.sigmoid(self.decay*(length-scal/length))

        # putting the line in the format [1, 3, imsize, imsize]
        line13 = line.unsqueeze(0).expand(3, imsize, imsize).unsqueeze(0)

        # returning a copy of the drawing with the line
        drawing_copy = torch.tensor(current_drawing)
        output=drawing_copy*line13
        imshow(output)
        return output


# main function
def run(input_img, n_lines, n_epoch=10):
    cnn = CNNFeatureExtractor()

    # retrain the model on small datasets containing hand drawn sketches NOT YET
    # model.fine_tune_model(["url_sketches"])
    for param in cnn.model.parameters():  # the cnn feature extractor has already been trained, we freeze its parameters
        param.requires_grad = False
    cnn.add_comparison_loss(input_img)
    drawer = Drawer(input_img)
    for k in range(n_lines):
        print("Drawing line number %i" % k)
        drawer.run_segment_optimizer(cnn, n_epoch)


