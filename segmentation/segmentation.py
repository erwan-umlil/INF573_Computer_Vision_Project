from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np


# We can also use T.Pad
def expand2square(pil_img, background_color):
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

# Define the helper function
def decode_segmap(image, nc=21):
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

def threshold_mask(mask, threshold):
    mask[mask > threshold] = 255



fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

img_path = 'images/car.png'
img = Image.open(img_path)
original_size = img.size
img_sq = expand2square(img, (0,0,0))
squared_size = img_sq.size[0]
#plt.imshow(img)
#plt.show()
print(img_sq.size)


trf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
inp = trf(img_sq).unsqueeze(0)
print(inp.shape)

#plt.imshow(inp[0].transpose(0, 1).transpose(1, 2))
#plt.show()

out = fcn(inp)['out']
print(out.shape)


om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print(om.shape)
print(np.unique(om))

rgb = decode_segmap(om)
#plt.imshow(rgb)
#plt.show()
#print(rgb.shape)

inv_trf = T.Compose([T.ToPILImage(), T.Resize(squared_size), T.ToTensor(), T.Normalize(mean = [-0.485, -0.456, -0.406], std=[1,1,1]), T.Normalize(mean=[0,0,0], std = [1/0.229, 1/0.224, 1/0.225])])
mask = inv_trf(rgb).transpose(0, 1).transpose(1, 2)
#threshold_mask(mask, 0.5)
print(mask.shape)

x, y, _ = mask.shape
mask = mask[y//2 - original_size[1]//2:y//2 + original_size[1]//2, x//2 - original_size[0]//2:x//2 + original_size[0]//2]
print(T.ToTensor()(img).transpose(0, 1).transpose(1, 2).shape, mask.shape)
final = T.ToTensor()(img).transpose(0, 1).transpose(1, 2) + mask
plt.imshow(final)
plt.show()
