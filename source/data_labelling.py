import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from utility import *

config = read_configuration()
logging.basicConfig(
    filename = config["log_filename"].replace("$$timestamp$$", datetime.now().strftime("%Y_%m_%d_%H_%M_%S")).replace("$$module_name$$", "Data Labelling"),
    level = logging.INFO,
    format='[%(asctime)s] - %(levelname)s -> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def create_labels():
    logging.info("Started creating labels in training data.")
    try:
        # Read the file
        train_df = pd.read_csv(os.path.join(config["dataset_path"]["training_data"], "meta_data.csv"))
    except Exception as e:
        logging.error(f"Failed to read the training data file due to {e}")
    try:
        # Apply the label rules
        train_df = apply_flare_thresholds(train_df, config["flux_column_name"])
    except Exception as e:
        logging.error(f"Failed to apply the flux thresholds in training data due to {e}")

    logging.info("Completed creating labels in training data.")

    logging.info("Started creating labels in test data.")
    
    try:
        # Read the file
        test_df = pd.read_csv(os.path.join(config["dataset_path"]["test_data"], "meta_data.csv"))
    except Exception as e:
        logging.error(f"Failed to read the test data file due to {e}")
    try:
        # Apply the label rules
        test_df = apply_flare_thresholds(test_df, config["flux_column_name"])
    except Exception as e:
        logging.error(f"Failed to apply the flux thresholds in test data due to {e}")
    
    logging.info("Completed creating labels in test data.")
    return train_df, test_df

def create_binary_target(df):
    logging.info("Started creating the binary targets.")
    for binary_target_class in config["binary_target_mapping"].keys():
        try:
            df[binary_target_class] = np.where(
                df["flare_class"].isin(config["binary_target_mapping"][binary_target_class]),
                1,
                0
            )

            logging.debug(f"The distribution of positive and negative class for {binary_target_class} is {df[binary_target_class].value_counts()}")
        except Exception as e:
            logging.error(f"Failed to create {binary_target_class} due to {e}")
    logging.info("Completed creating the binary targets.")
    return df
