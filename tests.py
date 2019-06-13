from optim_draw import *


def test_img_and_cnn():
    input_img = image_loader("./Images/boat.jpg")
    print(input_img.size())
    imshow(input_img)
    cnn = CNNFeatureExtractor()
    imshow(cnn.model(input_img))


def test_line_drawer():
    blank_img = torch.ones([1, 3, imsize, imsize], dtype=torch.float32)
    line_drawer = LineDrawer()
    drawn = line_drawer.forward(blank_img)
    imshow(drawn)


def test_overall(names_list):
    for name in names_list:
        _ = run(image_loader("./Images/"+name+".jpg"), 50, 50, unblur=True, save=True, save_title=name)


# test_img_and_cnn()
#test_line_drawer()
test_overall(["boat","test"])
