from optim_draw import *


def test_img_and_cnn():
    input_img = image_loader("./Images/dancer.jpg")
    print(input_img.size())
    imshow(input_img)
    cnn = CNNFeatureExtractor()
    imshow(cnn.model(input_img))


test_img_and_cnn()
