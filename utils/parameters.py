import argparse
from typing import Any
import os
WIDTH_IMAGE_SIZE  = 512
HEIGHT_IMAGE_SIZE = 512
GROUPS_MALIGNANT_SKIN_CANCER= ['MEL', 'BCC', 'SCC', 'MALO']
GROUPS_CLASS_LABEL = ['malignant', 'benign']
SUB_GROUPS_CLASS_LABEL = ['BCC', 'BENO', 'Lentigo', 'MEL', 'Melanosis', 'NV']
#------------------------------------------------------------------------------------------------------------------------------------------
def load_sub_class_weights():
    sub_class_weights = {}
    sub_class_weights['BCC'] = 2
    sub_class_weights['NV'] = 1
    sub_class_weights['BENO'] = 1
    sub_class_weights['Melanosis'] = 1
    return sub_class_weights
#-------------------------------------------------------------------------------------------------------------------------------------------
def load_class_weights():
    class_weights = {}
    class_weights['malignant'] = 3
    class_weights['benign'] = 2
    return class_weights
#-------------------------------------------------------------------------------------------------------------------------------------------
def initRunFolders(root, path):
    i = 0
    savepath = root + "/" + path
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    while True:
        runfolders = root + "/" + path + "/exp" + str(i)
        if not os.path.exists(runfolders):
            os.mkdir(runfolders)
            break
        else:
            i = i + 1
    return runfolders
#-------------------------------------------------------------------------------------------------------------------------------------------
def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg(
        "-r",
        "--resume",
        default=False,
        type=bool,
        help="continue to training",
        required=False,
    )

    arg(
        "-d",
        "--device",
        nargs="+",
        default=[0],
        type=int,
        help="use device to train",
        required=False,
    )
    arg(
        "-m",
        "--mixed_precision",
        default=True,
        type=bool,
        help="use mixed_precision or not",
        required=False,
    )
    arg(
        "-l",
        "--log",
        default="wandb",
        type=str,
        help="use logs",
        required=False,
    )
    arg(
        "-s",
        "--batch_size",
        default=5,
        type=int,
        help="batch size",
        required=False,
    )
    arg(
        "-rd",
        "--rate_decay",
        default=0.999,
        type=float,
        help="rate decay",
        required=False
    )
    arg(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="number of epoch",
        required=False,
    )
    arg(
        "-lr",
        "--learning_rate",
        default=0.0001,
        type=float,
        help="learning rate",
        required=False,
    )

    return parser.parse_args()