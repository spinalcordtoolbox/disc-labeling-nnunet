"""
This script is based on https://github.com/ivadomed/utilities/blob/main/dataset_conversion/convert_bids_to_nnUNetV2.py

Converts BIDS-structured dataset to the nnUNetv2 dataset format. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Naga Karthik, Jan Valosek, Théo Mathieu modified by Nathan Molinier
"""
import argparse
import pathlib
from pathlib import Path
import json
import os
from collections import OrderedDict
from loguru import logger
import numpy as np
from progress.bar import Bar

from utils import CONTRAST, get_img_path_from_label_path, fetch_subject_and_session, fetch_contrast
from image import Image


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet (Required)')
    parser.add_argument('--dataset-name', '-dname', default='MyDataset', type=str,
                        help='Specify the task name. (Default=MyDataset)')
    parser.add_argument('--dataset-number', '-dnum', default=501, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. (Default=501)')
    parser.add_argument('--seed', default=99, type=int,
                        help='Seed to be used for the random number generator split into training and test sets. (Default=99)')
    parser.add_argument('--registered', default=False, type=bool,
                        help='Set this variable to True if all the modalities/contrasts are available and corregistered for every subject (Default=False)')
    parser.add_argument('--one-class', default=False, type=bool,
                        help='Set this variable to True if all the discs are part of the same class (Default=False)')
    return parser


def convert_subjects(list_labels, path_out_images, path_out_labels, channel_dict, DS_name, counter_indent=0, one_class=False):
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

    # Init progression bar
    bar = Bar('Convert data ', max=len(list_labels))

    for label_path in list_labels:
        img_path = get_img_path_from_label_path(label_path)
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f'Error while loading subject\n {img_path} or {label_path} might not exist --> skipping subject')
        else:
            # Increment counter for every path --> different from nnunet conventional use where the same number is the same for every subject (but need full registration)
            # TODO: fix this case to keep the subject number for each contrast
            counter+=1

            # Extract information from the img_path
            sub_name, sessionID, filename, modality = fetch_subject_and_session(img_path)

            # Extract contrast from channel_dict
            if 'multi_contrasts' in channel_dict.keys():
                contrast = 'multi_contrasts'
            else:
                contrast = fetch_contrast(img_path) 

            # Create new nnunet paths
            nnunet_label_path = os.path.join(path_out_labels, f"{DS_name}-{sub_name}_{counter:03d}.nii.gz")
            nnunet_img_path = os.path.join(path_out_images, f"{DS_name}-{sub_name}_{counter:03d}_{channel_dict[contrast]:04d}.nii.gz")

            # Load and reorient image and label to RSP
            label = Image(label_path).change_orientation('RSP')
            img = Image(img_path).change_orientation('RSP')

            # Create new discs masks with spots instead of single points
            label_mask, max_label = create_disc_mask(label, one_class=one_class)
            
            # Update number of class
            if one_class:
                nb_class = 1
            elif max_label > nb_class:
                nb_class = max_label

            # Save images
            label_mask.save(nnunet_label_path)
            img.save(nnunet_img_path)
        # Plot progress
        bar.suffix  = f'{list_labels.index(label_path)+1}/{len(list_labels)}'
        bar.next()
    bar.finish()
    return counter, nb_class+1 # +1 to add the background


def create_disc_mask(label, one_class, radius=2):
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
    for coord in label.getNonZeroCoordinates(sorting='value'):
        if coord[-1] <= 25: # Remove labels superior to 25, especially 49 and 50 that correspond to the pontomedullary groove (49) and junction (50)
            if coord[-1] > max_label: # Extract max label
                max_label = coord[-1]
            x0, y0, z0, value = coord

            x, y, z = np.mgrid[0:nx:1,0:ny:1,0:nz:1]
            mask = ((px*(x-x0))**2 + (py*(y-y0))**2 + (pz*(z-z0))**2) <= radius**2
            new_label[mask] = value if not one_class else 1

    
    # Update label with new created mask
    label.data = new_label
    
    return label, max_label


def main():
    parser = get_parser()
    args = parser.parse_args()
    DS_name = args.dataset_name
    path_out = Path(os.path.join(os.path.abspath(os.path.expanduser(args.path_out)),
                                 f'Dataset{args.dataset_number:03d}_{args.dataset_name}'))
    # Read json file and create a dictionary
    with open(args.config, "r") as file:
        config = json.load(file)

    # To use channel dict with different modalities/contrasts, images need to be corregistered and all modalities/contrasts
    # need to be available.
    channel_dict = {}
    if args.registered:
        for i, contrast in enumerate(CONTRAST[config['CONTRASTS']]):
            channel_dict[contrast] = i
    else:
        channel_dict['multi_contrasts'] = 0

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
                                                  DS_name=DS_name,
                                                  one_class=args.one_class)

    # Convert testing subjects to nnunet format
    counter_test, nb_class_test = convert_subjects(list_labels=test_labels,
                                                path_out_images=path_out_imagesTs,
                                                path_out_labels=path_out_labelsTs,
                                                channel_dict=channel_dict,
                                                DS_name=DS_name,
                                                counter_indent=counter_train,
                                                one_class=args.one_class)

    logger.info(f"Number of training and validation subjects: {counter_train}")
    logger.info(f"Number of test subjects: {counter_test-counter_train}")

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

    json_dict['labels'] = {"background": 0}
    
    # Adding discs number as individual classes
    for num_disc in range(max(nb_class_train, nb_class_test)):
        json_dict['labels'][f'disc_{num_disc+1}'] = num_disc+1

    json_dict["numTraining"] = counter_train

    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"
    json_dict["overwrite_image_reader_writer"] = "SimpleITKIO"

    # create dataset.json
    json.dump(json_dict, open(os.path.join(path_out, "dataset.json"), "w"), indent=4)

if __name__ == '__main__':
    main()