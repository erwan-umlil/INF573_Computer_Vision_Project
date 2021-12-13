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
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int, required=True)
    parser.add_argument('--quantization_flag', help='specify if to perform color quantization training', type=bool, default=False)
    parser.add_argument('--mode', help='task to be done', default='paint2image')
    args = parser.parse_args()

    # Segmentation
    print('Segmentation...')
    remove_labels = args.remove.split(',')
    remove_labels = list(map(int, remove_labels))
    f, m = segment(args.input_dir + '/' + args.input_name, remove_labels)
    name = args.input_name.split('/')[-1][:-4]
    f.save(args.output_segmentation + "/s_" + name + '.png', "PNG")
    m.save(args.output_segmentation + "/s_" + name + "_mask.png", "PNG")
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
        if (opt.paint_start_scale < 1) | (opt.paint_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            if ref.shape[3] != real.shape[3]:
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            N = len(reals) - 1
            n = opt.paint_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            if opt.quantization_flag:
                opt.mode = 'paint_train'
                dir2trained_model = functions.generate_dir2save(opt)
                # N = len(reals) - 1
                # n = opt.paint_start_scale
                real_s = imresize(real, pow(opt.scale_factor, (N - n)), opt)
                real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
                real_quant, centers = functions.quant(real_s, opt.device)
                plt.imsave('%s/real_quant.png' % dir2save, functions.convert_image_np(real_quant), vmin=0, vmax=1)
                plt.imsave('%s/in_paint.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)
                in_s = functions.quant2centers(ref, centers)
                in_s = imresize(in_s, pow(opt.scale_factor, (N - n)), opt)
                # in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
                # in_s = imresize(in_s, 1 / opt.scale_factor, opt)
                in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
                plt.imsave('%s/in_paint_quant.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)
                if (os.path.exists(dir2trained_model)):
                    # print('Trained model does not exist, training SinGAN for SR')
                    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
                    opt.mode = 'paint2image'
                else:
                    train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, opt.paint_start_scale)
                    opt.mode = 'paint2image'
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.paint_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)

    print('Done!')
