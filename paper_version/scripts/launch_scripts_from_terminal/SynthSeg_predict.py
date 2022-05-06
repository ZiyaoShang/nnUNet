"""This script enables to launch predictions with SynthSeg from the terminal."""

# print information
print('\n')
print('SynthSeg prediction')
print('\n')

# python imports
import os
import sys
import numpy as np
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
parser.add_argument("--out_posteriors", type=str, default=None, dest="path_posteriors",
                    help="path where to save the posteriors. Must be the same type as path_images (path to a single "
                         "image or to a folder)")
parser.add_argument("--out_volumes", type=str, default=None, dest="path_volumes",
                    help="path to a csv file where to save the volumes of all ROIs for all patients")
args = vars(parser.parse_args())

# default parameters
path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/SynthSeg_segmentation_labels.npy')
path_model = "/home/turja/PV-SynthSeg_training/dice_013.h5"
print(path_model)
args['segmentation_label_list'] = path_label_list
args['path_model'] = path_model
args['sigma_smoothing'] = 0.5
args['keep_biggest_component'] = True
args['aff_ref'] = 'FS'

# call predict
predict(**args)
