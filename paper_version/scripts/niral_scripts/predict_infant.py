"""This script enables to launch predictions with SynthSeg from the terminal."""

# print information
print('\n')
print('SynthSeg prediction')
print('\n')

# python imports
import os
import sys
from argparse import ArgumentParser

# add main folder to python path and import ./SynthSeg/predict.py
synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
from SynthSeg.predict import predict


# parse arguments
parser = ArgumentParser()
parser.add_argument("path_images", type=str, help="images to segment. Can be the path to a single image or to a folder")
parser.add_argument("path_segmentations", type=str, help="path where to save the segmentations. Must be the same type "
                                                         "as path_images (path to a single image or to a folder)")
parser.add_argument("path_model", type=str, help="Trained model")
parser.add_argument("--out_posteriors", type=str, default=None, dest="path_posteriors",
                    help="path where to save the posteriors. Must be the same type as path_images (path to a single "
                         "image or to a folder)")
parser.add_argument("--out_volumes", type=str, default=None, dest="path_volumes",
                    help="path to a csv file where to save the volumes of all ROIs for all patients")
parser.add_argument("--aff_ref", type=str, default='identity')
args = vars(parser.parse_args())

# default parameters
path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/SynthSeg_segmentation_labels.npy')
# args['segmentation_label_list'] = path_label_list
# args['segmentation_label_list'] = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60, 61]
args['segmentation_label_list'] = [0, 14, 15, 16, 24, 85, 86, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13,
                    17, 18, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]
args['sigma_smoothing'] = 0.5
args['keep_biggest_component'] = True
args['aff_ref'] = 'identity'
args['gt_folder'] = '/home/ziyaos/ziyao_data/new_labels/DCAN_roi3_WM'
args['activation'] = 'elu'
# args['padding'] = 43
# call predict
predict(**args)


