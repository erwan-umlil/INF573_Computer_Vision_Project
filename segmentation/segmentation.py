from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
import cv2 as cv


# We can also use T.Pad
def expand2square(pil_img, background_color):
    """
    Expand pil_img to a square image with background_color-padding
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def square2original(img, original_size):
    """
    Extract an image with the original_size from img (inverse transformation of expand2square)
    """
    width, height = original_size
    if width == height:
        return img
    elif width > height:
        return img[(width - height) // 2:(width - height) // 2 + height, :]
    else:
        return img[:, (height - width) // 2:(height - width) // 2 + width]

def threshold_mask(mask, threshold):
    """
    Transform mask into a black and white image according to threshold's value
    """
    # img numpy array
    gray_img = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    gray_img[gray_img>threshold] = 255
    gray_img[gray_img<threshold] = 0
    return gray_img

def decode_segmap(image, nc=21):
    """
    Return a RGB image corresponding to image, which is a raw output of the network
    """
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb



# Load resnet101 pretrained model
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

# Load image
name = 'car.png'
#name = 'bird.png'
img_path = 'images/' + name
img = Image.open(img_path)
original_size = img.size
img_sq = expand2square(img, (0,0,0))
squared_size = img_sq.size[0]


# Transform image to fit with resnet input size
trf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
inp = trf(img_sq).unsqueeze(0)

# Forward through the network
out = fcn(inp)['out']

# Analysis of the output
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
rgb = decode_segmap(om)

# Transformation of the output mask
inv_trf = T.Compose([T.ToPILImage(), T.Resize(squared_size), T.ToTensor(), T.Normalize(mean = [-0.485, -0.456, -0.406], std=[1,1,1]), T.Normalize(mean=[0,0,0], std = [1/0.229, 1/0.224, 1/0.225])])
mask = inv_trf(rgb).transpose(0, 1).transpose(1, 2)
mask = square2original(mask, original_size).numpy()
mask = threshold_mask(mask, 0.1)

# Apply the mask to the original picture
final = T.ToTensor()(img).transpose(0, 1).transpose(1, 2)
final[mask>0.1,:,:] = 1

# Show the result
plt.imshow(final)
plt.show()

f = T.ToPILImage()(final.transpose(1, 2).transpose(0, 1))
f.save("preprocessed_images/s_" + name,"PNG")
