from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse


def diffuse_one_time_step(res, mask, K, dx, dy):
    """Perform the heat equation on one time step"""
    n, m = mask.shape
    for i in range(1, n-1):
        for j in range(1, m-1):
            if mask[i,j] < 0.5: # not in a hole
                continue
            diff = K * ((res[i-1,j] - 2*res[i,j] + res[i+1,j]) / (dx**2) + (res[i,j-1] - 2*res[i,j] + res[i,j+1]) / (dy**2))
            res[i,j] += diff.astype(int)

def heat_inpaint(img, mask, K, dx, dy, epochs=100):
    """Perform inpainting using heat equation"""
    mask /= np.max(mask)
    res = img.copy() - 255 * mask[:,:,np.newaxis].astype(int)
    for _ in tqdm(range(epochs)):
        diffuse_one_time_step(res, mask, K, dx, dy)
    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='path to the input image')
    parser.add_argument('--mask', type=str, help='path to the input mask')
    # parser.add_argument('--output', type=str, default='', help='path where the output image will be saved')
    parser.add_argument('--K', type=float, default=5e-1, help='diffusion coefficient')
    parser.add_argument('--dx', type=float, default=1, help='spatial step dx')
    parser.add_argument('--dy', type=float, default=1, help='spatial step dy')

    args = parser.parse_args()
    img = Image.open(args.image).convert('RGB')
    img = np.array(img)
    mask = Image.open(args.mask).convert('L')
    mask = np.array(mask).astype(float)

    res = heat_inpaint(img, mask, args.K, args.dx, args.dy)

    plt.imshow(res)
    plt.show()
