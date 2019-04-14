from optim_draw import *


def test_img_and_cnn():
    input_img = image_loader("./Images/dancer.jpg")
    print(input_img.size())
    imshow(input_img)
    cnn = CNNFeatureExtractor()
    imshow(cnn.model(input_img))


def test_line_drawer():
    blank_img = torch.zeros([1, 3, imsize, imsize], dtype=torch.int32)
    line_drawer = LineDrawer()
    line_drawer.forward(blank_img)


test_img_and_cnn()
test_line_drawer()
