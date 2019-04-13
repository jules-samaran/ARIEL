from optim_draw import *

input_img = image_loader("./Images/dancer.jpg")
imshow(input_img)
cnn = CNNFeatureExtractor()
imshow(cnn.model(input_img))

