import argparse

from segmentation.segmentation import segment
from edge_connect.main import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', help='image from which we remove an object', required=True)
    parser.add_argument('--remove', type=str, default='15', help='labels of objects to remove, e.g. 1,2,3,4')
    parser.add_argument('--path', '--checkpoints', type=str, default='edge_connect/checkpoints/places2', help='model checkpoints path (default: edge_connect/checkpoints/places2)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model', default=4)
    parser.add_argument('--input', type=str, help='path where we store the segmented image')
    parser.add_argument('--mask', type=str, help='path where we store mask file')
    parser.add_argument('--edge', type=str, help='path to an edge file')
    parser.add_argument('--output', type=str, help='path to the output directory (default: output/edge_connect)', default='output/edge_connect')

    args = parser.parse_args()

    # Segmentation
    print('Segmentation...')
    remove_labels = args.remove.split(',')
    remove_labels = list(map(int, remove_labels))
    f, m = segment(args.input_img, remove_labels)
    name = args.input_img.split('/')[-1][:-4]
    f.save(args.input, "PNG")
    m.save(args.mask, "PNG")

    # Inpainting using edge-connect
    main(2, args)