from optim_draw import *
from model_encoder import *


def get_1line_image(x1, y1, x2, y2, sharp=False):
    drawing = torch.ones([1, 3, imsize, imsize], dtype=torch.float32)
    start_point = torch.tensor([x1, y1], dtype=torch.float32)
    end_point = torch.tensor([x2, y2], dtype=torch.float32)
    width = torch.tensor(1 * imsize/64, dtype=torch.float32)
    decay = torch.tensor(1.0/(imsize/64), dtype=torch.float32)
    if sharp:
        width = torch.tensor(0, dtype=torch.float32)
        decay = 20.0/(imsize/64)
    length = torch.dist(start_point, end_point)

    i_values = torch.tensor([[i for j in range(imsize)] for i in range(imsize)], dtype=torch.float32)
    j_values = torch.tensor([[j for j in range(imsize)] for i in range(imsize)], dtype=torch.float32)

    start_point_x = start_point[0].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize, imsize)
    start_point_y = start_point[1].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize, imsize)
    end_point_x = end_point[0].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize, imsize)
    end_point_y = end_point[1].unsqueeze(0).expand(imsize).unsqueeze(0).expand(imsize, imsize)

    # determinant for line width
    det = (j_values - start_point_x) * (end_point_y - start_point_y) - \
          (i_values - start_point_y) * (end_point_x - start_point_x)

    # scalar product to test belonging to the segment
    scal = (j_values - start_point_x) * (end_point_x - start_point_x) + \
           (i_values - start_point_y) * (end_point_y - start_point_y)

    # combining the above tensors to obtain the line, using sigmoids for differentiability
    line = torch.ones([imsize, imsize]) - \
           torch.sigmoid(decay * (det / length + width / 2)) * torch.sigmoid(
        decay * (width / 2 - det / length)) \
           * torch.sigmoid(decay * scal / length) * torch.sigmoid(decay * (length - scal / length))

    # putting the line in the format [1, 3, imsize, imsize]
    line13 = line.unsqueeze(0).expand(3, imsize, imsize).unsqueeze(0)

    # returning a copy of the drawing with the line
    drawing = drawing * line13
    return drawing


ref_image = get_1line_image(32, 16, 32, 48)
imshow(ref_image)
cnn = CNNFeatureExtractor()


for param in cnn.model.parameters():  # the cnn feature extractor has already been trained, we freeze its parameters
    param.requires_grad = False
cnn.add_comparison_loss(ref_image)
losses = []
for k in range(30):
    drawing = get_1line_image(32 + k, 16, 32 + k, 48)
    losses.append(cnn.comparison_loss(cnn.model(drawing)))

plt.figure()
plt.plot(np.arange(len(losses)), losses)
plt.xlabel("Distance to the reference line")
plt.ylabel("Loss")
plt.show()
