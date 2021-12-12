from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torchvision.transforms as T
import numpy as np
import cv2 as cv
import argparse


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

def decode_segmap(image, labels_to_remove, nc=21):
    """
    Return a RGB image corresponding to image, which is a raw output of the network
    """
    assert(max(labels_to_remove) < nc)
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
    for l in labels_to_remove:
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def preprocess_image(image):
    """Preprocess input image to have a size 4*n and resize it"""
    img = image.crop((0, 0, 4*(image.size[0]//4), 4*(image.size[1]//4)))
    if img.size[0] >= img.size[1]:
        img = img.resize((640, int(img.size[1] * 640 / img.size[0])))
    else:
        img = img.resize((int(img.size[0] * 640 / img.size[1]), img.size[1]))
    return img


def segment(img_path, output_path, remove_labels):
    """Segment the input image, erase the objects described by remove_labels and save the result and the mask"""
    name = img_path.split('/')[-1][:-4]
    # Load resnet101 pretrained model
    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

    img = Image.open(img_path).convert('RGB')
    img = preprocess_image(img)
    original_size = img.size
    img_sq = expand2square(img, (0,0,0))
    squared_size = img_sq.size[0]

    # Transform image to fit with resnet input size
    trf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    inp = trf(img_sq).unsqueeze(0)

    # Forward through the network
    out = fcn(inp)['out']

    # Analyze the output
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om, remove_labels)

    # Transform the output mask
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

    # Save the preprocessed image and the mask
    f = T.ToPILImage()(final.transpose(1, 2).transpose(0, 1))
    f.save(output_path + "s_" + name + '.png', "PNG")
    m = Image.fromarray(np.uint8(cm.gist_earth(mask) * 255))
    m.save(output_path + "s_" + name + "_mask.png", "PNG")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='', help='path to the input image')
    parser.add_argument('--output', type=str, default='segmented_images/', help='path where the output image will be saved')
    parser.add_argument('--remove', type=str, default='15', help='labels of objects to remove, e.g. 1,2,3,4')

    args = parser.parse_args()
    remove_labels = args.remove.split(',')
    remove_labels = list(map(int, remove_labels))

    segment(args.image, args.output, remove_labels)
