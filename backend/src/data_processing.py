import os
import argparse
import shutil
import numpy as np  
from typing import List
import glob
from utils import Logger, AppPath
from config.data_config import CatDog_Data

LOGGER = Logger(__file__)
LOGGER.log.info("Starting Data Preprocessing")

def create_training_data(version: str, source_dir: List[str], dest_dir: str, ratio: List[float]):
    """
    Create train/val/test data from raw data
    """

    LOGGER.log.info(f"Begin create train/val/tets data")
    LOGGER.log.info(f"Version: {version}")
    LOGGER.log.info(f"Source dir: {source_dir}")
    LOGGER.log.info(f"Destination dir: {dest_dir}")
    LOGGER.log.info(f"Ratio [train, val]: {ratio}")

    for cls in CatDog_Data.classes:
        os.makedirs(f"{dest_dir}/{version}/train/{cls}", exist_ok=True)
        os.makedirs(f"{dest_dir}/{version}/val/{cls}", exist_ok=True)
        os.makedirs(f"{dest_dir}/{version}/test/{cls}", exist_ok=True)

    for cls in CatDog_Data.classes:
        LOGGER.log.info(f"Processing {cls} ...")
        all_cls_files = []
        for source in source_dir:
            source_cls = f"{source}/{cls}"
            all_cls_files.extend(glob.glob(f"{source_cls}/*.jpg"))

        num_files = len(all_cls_files)
        LOGGER.log.info(f"Number of files of {cls}: {num_files}")

        all_cls_files = np.array(all_cls_files)
        shuffle_indices = np.random.permutation(num_files)

        num_train = int(num_files * ratio[0])
        num_val = int(num_files * ratio[1])
        # num_test = num_files - num_train - num_val

        train_files = all_cls_files[shuffle_indices][:num_train]
        val_files = all_cls_files[shuffle_indices][num_train:num_train + num_val]
        test_files = all_cls_files[shuffle_indices][num_train + num_val:]

        for file in train_files:
            shutil.copy(file, f"{dest_dir}/{version}/train/{cls}/{os.path.basename(file)}")
        for file in val_files:
            shutil.copy(file, f"{dest_dir}/{version}/val/{cls}/{os.path.basename(file)}")
        for file in test_files: 
            shutil.copy(file, f"{dest_dir}/{version}/test/{cls}/{os.path.basename(file)}")

    LOGGER.log.info(f"Finish create train/val/tets data")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, 
                        help="Version of the data, e.g. v1, as the name of the folder")
    parser.add_argument("--merge_collected", action='store_true', 
                        help="Merge collected data to raw data")
    parser.add_argument("--dest_dir", type=str, default=AppPath.TRAIN_DATA_DIR, 
                        help="Destination of directory to save the train/val/test data")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.6, 0.2], 
                        help="Ratio of train/val/test data")
    args = parser.parse_args()
    
    if os.path.exists(AppPath.TRAIN_DATA_DIR / args.version):
        shutil.rmtree(AppPath.TRAIN_DATA_DIR / args.version)
    
    source_dir = [AppPath.RAW_DATA_DIR]
    if args.merge_collected:
        source_dir += [AppPath.COLLECTED_DATA_DIR]

    create_training_data(args.version, source_dir, args.dest_dir, args.ratio)
    
