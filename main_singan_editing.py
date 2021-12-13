from PIL import Image
import numpy as np
import matplotlib

from segmentation.segmentation import segment
from diffusion.heat import heat_inpaint
from SinGAN.config import get_arguments
from SinGAN.SinGAN.manipulate import *
from SinGAN.SinGAN.training import *
from SinGAN.SinGAN.imresize import imresize
from SinGAN.SinGAN.imresize import imresize_to_shape
import SinGAN.SinGAN.functions as functions


if __name__ == "__main__":
    parser = get_arguments()
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--input_dir', help='input image dir', default='images')
    parser.add_argument('--output', type=str, default='output/SinGAN', help='path where the output image will be saved')
    parser.add_argument('--output_segmentation', type=str, default='output/segmentation', help='path where the segmented image and its mask will be saved')
    parser.add_argument('--remove', type=str, default='15', help='labels of objects to remove, e.g. 1,2,3,4')
    parser.add_argument('--K', type=float, default=5e-1, help='diffusion coefficient')
    parser.add_argument('--dx', type=float, default=1, help='spatial step dx')
    parser.add_argument('--dy', type=float, default=1, help='spatial step dy')
    parser.add_argument('--heat_epochs', type=int, default=100, help='epochs to diffuse heat equation')
    parser.add_argument('--ref_dir', help='heat edited dir', default='output/heat')
    parser.add_argument('--ref_name', help='heat edited image name', required=True)
    parser.add_argument('--editing_start_scale', help='editing injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='editing')
    args = parser.parse_args()

    # Segmentation
    print('Segmentation...')
    remove_labels = args.remove.split(',')
    remove_labels = list(map(int, remove_labels))
    _, m = segment(args.input_dir + '/' + args.input_name, args.output_segmentation, remove_labels)
    m.save(args.ref_dir + "/" + args.ref_name[:-4] + "_mask.png", "PNG")

    # Inpainting using heat equation
    print('Naive inpainting using heat equation...')
    img = Image.open(args.output_segmentation + '/s_' + args.input_name).convert('RGB')
    img = np.array(img)
    mask = Image.open(args.output_segmentation + '/s_' + args.input_name[:-4] + '_mask.png').convert('L')
    mask = np.array(mask).astype(float)
    res = heat_inpaint(img, mask, args.K, args.dx, args.dy, args.heat_epochs)
    matplotlib.image.imsave(args.ref_dir + '/' + args.ref_name, np.uint8(res))

    # SinGAN editing
    print('SinGAN editing...')
    opt = functions.post_config(args)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = args.output + '/' + args.input_name[:-4]
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.editing_start_scale < 1) | (opt.editing_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
            if ref.shape[3] != real.shape[3]:
                '''
                mask = imresize(mask, real.shape[3]/ref.shape[3], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize(ref, real.shape[3] / ref.shape[3], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
                '''
                mask = imresize_to_shape(mask, [real.shape[2],real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2],real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.editing_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.editing_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)
            #mask, real, out are Torch tensors of size :
            #mask  torch.Size([1, 3, 168, 250])
            #real  torch.Size([1, 3, 168, 250])
            #out  torch.Size([1, 3, 130, 193])
            #out = imresize_to_shape(out, real.shape, opt)
            out = (1-mask)*real+mask*out
            plt.imsave('%s/start_scale=%d_masked.png' % (dir2save, opt.editing_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)

    print('Done!')
