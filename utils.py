import argparse
import time
import numpy as np
import os
import tensorflow as tf

from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.run.run_training import merge_train
from nnunet.paths import default_plans_identifier
from nnunet.experiment_planning import nnUNet_plan_and_preprocess

from paper_version.ext.lab2im import utils
from paper_version.SynthSeg import brain_generator


def generate_dataset(num, merge_labels=None):

    # number of images to generate
    num_image = num

    # path of the input label map
    labels_dir = "/home/ziyaos/SSG_HDN/training_labels"

    # path where to save the generated image
    result_label = '/home/ziyaos/nnunet_ssg/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ssgtry/labelsTr'
    result_T1 = '/home/ziyaos/nnunet_ssg/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ssgtry/imagesTr'
    result_T2 = '/home/ziyaos/nnunet_ssg/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ssgtry/imagesTr'

    # path to the priors
    T1_means = "/home/ziyaos/SSG_HDN/T1merged/prior_means.npy"
    T1_stds = "/home/ziyaos/SSG_HDN/T1merged/prior_stds.npy"

    T2_means = "/home/ziyaos/SSG_HDN/T2merged/prior_means.npy"
    T2_stds = "/home/ziyaos/SSG_HDN/T2merged/prior_stds.npy"
    #
    generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2,  3,   4,   5,   7,   8,  10,  11,  12,  13, 17, 18,
                              21,  26,  28,  30,  31,  41,  42,  43,  44,  46,  47,  49,  50,  51,  52,  53,  54,  58,  60,
                              61,  62,  63])
    generation_classes = np.array([0, 1, 2, 3, 4, 5, 6, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 9, 22,
                                        23, 24, 25, 9, 27, 28, 29, 13, 31, 32, 33, 34, 35, 36, 37, 38, 39, 9, 41, 42])

    segmentation_labels = [0, 14, 15, 16, 170, 172, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 46,
                                47, 49, 50, 51, 52, 53, 54, 58, 60, 61]


    flipping = False  # whether to right/left flip the training label maps, this will take sided labels into account
    # (so that left labels are indeed on the left when the label map is flipped)
    scaling_bounds = .15  # the following are for linear spatial deformation, higher is more deformation
    rotation_bounds = 15
    shearing_bounds = .012
    translation_bounds = False  # here we deactivate translation as we randomly crop the training examples
    nonlin_std = 3.  # maximum strength of the elastic deformation, higher enables more deformation
    nonlin_shape_factor = .04  # scale at which to elastically deform, higher is more local deformation

    # bias field parameters
    bias_field_std = .5  # maximum strength of the bias field, higher enables more corruption
    bias_shape_factor = .025  # scale at which to sample the bias field, lower is more constant across the image

    # acquisition resolution parameters
    # the following parameters aim at mimicking data that would have been 1) acquired at low resolution (i.e. data_res),
    # and 2) upsampled to high resolution in order to obtain segmentation at high res (see target_res).
    # We do not such effects here, as this script shows training parameters to segment data at 1mm isotropic resolution
    data_res = None
    randomise_res = False
    thickness = None
    downsample = False
    blur_range = 1.03  # we activate this parameter, which enables SynthSeg to be robust against small resolution variations

    # no randomness when selecting the templetes for generation

    T1_generator = brain_generator.BrainGenerator(labels_dir, generation_labels=generation_labels, prior_means=T1_means,
                                  prior_stds=T1_stds, flipping=flipping, generation_classes=generation_classes,
                                  scaling_bounds=scaling_bounds,
                                  rotation_bounds=rotation_bounds,
                                  shearing_bounds=shearing_bounds,
                                  nonlin_std=nonlin_std,
                                  nonlin_shape_factor=nonlin_shape_factor,
                                  data_res=data_res,
                                  thickness=thickness,
                                  downsample=downsample,
                                  blur_range=blur_range,
                                  bias_field_std=bias_field_std,
                                  bias_shape_factor=bias_shape_factor,
                                  mix_prior_and_random=True,
                                  prior_distributions='normal',
                                  use_generation_classes=0.5)


    for i in range(num_image):
        start = time.time()
        im, lab = T1_generator.generate_brain()
        end = time.time()
        print('generation {0:d} took {1:.01f}s'.format(i, end - start))
        print(im.shape)
        # save output image and label map
        utils.save_volume(np.squeeze(im), T1_generator.aff, T1_generator.header,
                          os.path.join(result_T1, 'brain_' + str(i).rjust(3, '0') + '_0000.nii.gz'))
        utils.save_volume(np.squeeze(lab), T1_generator.aff, T1_generator.header,
                          os.path.join(result_label, 'brain_' + str(i).rjust(3, '0') + '.nii.gz'))

        print("Saved Output.")
    del T1_generator

    print("step two")
    # sequential selection
    T2_generator = brain_generator.BrainGenerator(result_label, generation_labels=generation_labels, prior_means=T2_means,
                                  prior_stds=T2_stds, generation_classes=generation_classes,
                                  data_res=data_res,
                                  thickness=thickness,
                                  downsample=downsample,
                                  blur_range=blur_range,
                                  bias_field_std=bias_field_std,
                                  bias_shape_factor=bias_shape_factor,
                                  mix_prior_and_random=True,
                                  prior_distributions='normal',
                                  use_generation_classes=0.5,
                                  flipping=False,
                                  apply_linear_trans=False,
                                  scaling_bounds=0,
                                  rotation_bounds=0,
                                  shearing_bounds=0,
                                  apply_nonlin_trans=False,
                                  nonlin_std=0,
                                  nonlin_shape_factor=0)

    label_names = utils.list_images_in_folder(result_label)
    for i in range(num_image):
        im, lab = T2_generator.generate_brain()
        print(im.shape)
        # save output image and label map
        brain_ind = label_names[i].split('/')[-1].split('.')[0].split('_')[-1]
        print(brain_ind)
        utils.save_volume(np.squeeze(im), T2_generator.aff, T2_generator.header,
                          os.path.join(result_T2, 'brain_' + str(i).rjust(3, '0') + '_0001.nii.gz'))

        print("Saved Output.")
    del T2_generator

    print("Generation finished, generated " + str(num_image) + " brains")

    # convert into large labels
    print("Step 3: convert labels")
    if merge_labels is not None:
        print("converting into " + str(len(merge_labels)) + " labels ")
        label_names = utils.list_images_in_folder(result_label)
        print(label_names)
        for lbmap in label_names:
            print(str(lbmap))
            volume, aff, header = utils.load_volume(path_volume=lbmap, im_only=False)
            for set in merge_labels:
                print(str(set))
                cvtTo = set[1]
                cvtArr = set[0]
                for label in cvtArr:
                    volume[volume == label] = cvtTo
            assert np.array_equal(np.unique(volume), np.arange(len(merge_labels))), str(np.unique(volume))
            print("saving...")
            utils.save_volume(np.squeeze(volume), aff, header, lbmap)
        print("convert finished")

# generate dataset.json
def generate_json():
    generate_dataset_json(output_file='/home/ziyaos/nnunet_ssg/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ssgtry/dataset.json',
                          imagesTr_dir='/home/ziyaos/nnunet_ssg/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ssgtry/imagesTr',
                          imagesTs_dir='/home/ziyaos/nnunet_ssg/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ssgtry/imagesTs',
                          modalities=('T1', 'T2'),
                          labels={0: 'background', 1: '1', 2: '2', 3: '3'},
                          dataset_name='Task001_ssgtry',
                          license='hands off!'
                          )

# train an nnUNet
def train(continue_training=False):
    '''
    :param continue_training: whether or not to load the latest model (which should always be the case except for he first epoch)
    '''
    parser = argparse.ArgumentParser()
    # parser.add_argument("network")
    # parser.add_argument("network_trainer")
    # parser.add_argument("task", help="can be task name or task id")
    # parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    args = parser.parse_args()

    args.network = '3d_fullres'
    args.network_trainer = 'nnUNetTrainerV2'
    args.task = 'Task001_ssgtry'
    args.fold = 1
    args.continue_training = continue_training

    merge_train(args)
