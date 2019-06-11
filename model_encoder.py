
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models



######################################################################
# if a GPU is available, running on it will be much faster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark=True

######################################################################
# Loading the Images

# desired size of the output image
#imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
imsize = 128

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


######################################################################
# Now, let's create a function that displays an image by reconverting a
# copy of it to PIL format and displaying the copy using
# ``plt.imshow``.

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()


def imshow(tensor, title=None, save=False):
    # displays an image which was in tensor format
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.axis('off')
    if save and title is not None:
        plt.savefig(title)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


######################################################################
# Additionally, VGG networks are trained on images with each channel
# normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# We will use them to normalize the image before sending it into the network.
#

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean=cnn_normalization_mean, std=cnn_normalization_std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# We define which layer we are going to extract encoded features from
content_layers_default = ['conv_4']


class CNNFeatureExtractor:
    def __init__(self):
        # importing the pretrained CNN
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
        self.model = model[:9]  # [:(i + 1)] we truncate all the layers after the fourth conv layer

    # retrain the model on small datasets containing hand drawn sketches
    def fine_tune_model(self, additional_datasets_url):
        pass

    def add_comparison_loss(self, input_img):
        self.comparison_loss = ContentLoss(self.model, input_img)


######################################################################
# Content Loss
class ContentLoss(nn.Module):

    def __init__(self, model, target_image):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = model(target_image).detach()

    def forward(self, input):
        loss = F.mse_loss(input, self.target)
        return loss
