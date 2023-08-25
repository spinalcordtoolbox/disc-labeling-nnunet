"""
This script is based on https://github.com/ivadomed/utilities/blob/main/dataset_conversion/convert_bids_to_nnUNetV2.py

Converts BIDS-structured dataset to the nnUNetv2 dataset format. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Usage example:
    python convert_bids_to_nnUNetv2.py --path-data ~/data/dataset --path-out ~/data/dataset-nnunet
                    --dataset-name MyDataset --dataset-number 501 --split 0.8 0.2 --seed 99

Naga Karthik, Jan Valosek, Théo Mathieu modified by Nathan Molinier
"""
import re
import argparse
import shutil
import pathlib
from pathlib import Path
import json
import os
from collections import OrderedDict
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np

from utils import CONTRAST, get_img_path_from_label_path, fetch_subject_and_session
from image import Image


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet (Required)')
    parser.add_argument('--dataset-name', '-dname', default='MyDataset', type=str,
                        help='Specify the task name. Example: MyDataset')
    parser.add_argument('--dataset-number', '-dnum', default=501, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
    parser.add_argument('--seed', default=99, type=int,
                        help='Seed to be used for the random number generator split into training and test sets.')
    return parser


def convert_subject(list_labels, path_out_images, path_out_labels, channel_dict, DS_name, counter_indent=0):
    """Function to get image from original BIDS dataset modify if needed and place
        it with a compatible name in nnUNet dataset.

    Args:
        list_labels (list): List containing the paths of training/testing labels in the nnUNetv2 format.
        path_out_images (str): path to the images directory in the new dataset (test or train).
        path_out_labels (str): path to the labels directory in the new dataset (test or train).
        channel_dict (dict): Association dictionary between MRI contrasts and integer values compatible with nnUNet documentation (ex: T1w = 1, T2w = 2, FLAIR = 3).
        DS_name (str): Dataset name.
        counter_indent (int): indent for file numbering.

    Returns:
        counter (int): Last file number used
        nb_class (int): Maximum number of class

    """
    nb_class = 0
    counter = counter_indent
    for label_path in list_labels:
        img_path = get_img_path_from_label_path(path)
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f'Error while loading subject\n {img_path} or {label_path} might not exist --> skipping subject')
        else:
            # Increment counter for every path --> different from nnunet conventional use where the same number is the same for every subject (but need full registration)
            counter+=1

            # Extract information from the img_path
            sub_name, sessionID, filename, modality = fetch_subject_and_session(img_path)
            contrast = fetch_contrast(img_path)

            # Create new nnunet paths
            nnunet_label_path = os.path.join(path_out_labels, f"{DS_name}-{sub_name}_{counter:03d}.nii.gz")
            nnunet_img_path = os.path.join(path_out_images, f"{DS_name}-{sub_name}_{counter:03d}_{channel_dict[contrast]:04d}.nii.gz")

            # Load and reorient image and label to RSP
            label = Image(label_path).change_orientation('RSP')
            img = Image(img_path).change_orientation('RSP')

            # Create new discs masks with spots instead of single points
            label, max_label = create_disc_mask(label)

            # Update number of class
            if max_label > nb_class:
                nb_class = max_label

            # Extract number of discs labels
            label.getNonZeroCoordinates(sorting='value')[-1]

            # Save images
            label.save(nnunet_label_path)
            img.save(nnunet_img_path)
    return counter, nb_class+1 # +1 to add the background


def create_disc_mask(label):
    """
    Transform discs labels into discs masks with bigger spherical spots
    :param label: Image object of disc labels
    """
    
    # Create new numpy array
    new_label = np.zeros_like(label.data)

    # Get label dimensions
    nx, ny, nz, nt, px, py, pz, pt = label.dim

    # Loop in labels list
    max_label = 0
    radius = 10
    for coord in label.getNonZeroCoordinates(sorting='value'):
        if coord[-1] <= 25: # Remove labels superior to 25, especially 49 and 50 that correspond to the pontomedullary groove (49) and junction (50)
            if coord[-1] > max_label: # Extract max label
                max_label = coord[-1]
            x0, y0, z0, value = coord

            x, y, z = np.mgrid[0:nx:1]*px, np.mgrid[0:ny:1]*py, np.mgrid[0:nz:1]*pz
            r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
            r[r > radius] = 0
            new_label += r
    return new_label, max_label


def main():
    parser = get_parser()
    args = parser.parse_args()
    DS_name = args.dataset_name
    path_out = Path(os.path.join(os.path.abspath(os.path.expanduser(args.path_out)),
                                 f'Dataset{args.dataset_number:03d}_{args.dataset_name}'))
    # Read json file and create a dictionary
    with open(args.config, "r") as file:
        config = json.load(file)

    # Init channel dict
    channel_dict = {}
    for i, contrast in enumerate(CONTRAST[config[CONTRASTS]]):
        channel_dict[contrast] = i

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    train_labels = config['TRAINING'] + config['VALIDATION']
    test_labels = config['TESTING']

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # Convert training and validation subjects to nnunet format
    counter_train, nb_class_train = convert_subjects(list_labels=train_labels,
                                                  path_out_images=path_out_imagesTr,
                                                  path_out_labels=path_out_labelsTr,
                                                  channel_dict=channel_dict,
                                                  DS_name=DS_name)

    # Convert testing subjects to nnunet format
    counter_test, nb_class_test = convert_subjects(list_labels=test_labels,
                                                path_out_images=path_out_imagesTs,
                                                path_out_labels=path_out_labelsTs,
                                                channel_dict=channel_dict,
                                                DS_name=DS_name,
                                                counter_indent=counter_train)

    logger.info(f"Number of training and validation subjects: {counter_train}")
    logger.info(f"Number of test subjects: {counter_test}")

    # c.f. dataset json generation
    # In nnUNet V2, dataset.json file has become much shorter. The description of the fields and changes
    # can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson
    # this file can be automatically generated using the following code here:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()

    # The following keys are the most important ones.
    """
    channel_names:
        Channel names must map the index to the name of the channel. For BIDS, this refers to the contrast suffix.
        {
            "0": "FLAIR",
            "1": "T1w",
            "2": "T2",
            "3": "T2w"
        }
    Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based 
        training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training: 
        https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    """

    json_dict['channel_names'] = {v: k for k, v in channel_dict.items()}

    json_dict['labels'] = {
        "background": 0,
        f"{contrast}": 1,
    }

    json_dict["numTraining"] = train_ctr + 1
    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"
    json_dict["overwrite_image_reader_writer"] = "SimpleITKIO"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    main()