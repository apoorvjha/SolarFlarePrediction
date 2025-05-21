import os
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import logging
from utility import *
from tqdm import tqdm

config = read_configuration()
logging.basicConfig(
    filename = config["log_filename"].replace("$$timestamp$$", datetime.now().strftime("%Y_%m_%d_%H_%M_%S")).replace("$$module_name$$", "Data Preprocessing"),
    level = logging.INFO,
    format='[%(asctime)s] - %(levelname)s -> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class SDOBenchmarkDataset:
    def __init__(
            self,
            base_path,
            data,
            data_format = "channels_last"
        ):
        self.base_path = base_path
        self.data = data
        self.data_format = data_format

        # Define the timesteps to pick the magnetograms (aka. channels)
        self.channels = ["magnetogram"]
        self.timesteps = [0, 7*60, 10*60+30, 11*60+50]

        self.X, self.Y = self.load_data(self.data)
        logging.info("Dataset object instantiated!")

    def load_data(self, meta_data_df):
        logging.info("Starting ingesting the images and labels data.")
        X = []
        Y = []
        logging.info("Iterating over the metadata.")
        for idx, row in tqdm(meta_data_df.iterrows(), total = meta_data_df.shape[0]):
            images = np.zeros((4, 256, 256, 1))
            label = []
            sample_id = row["id"]
            chunks = sample_id.split("_")
            ar_number = chunks[0]
            date_time = datetime.strptime("_".join(chunks[1 : 7]), "%Y_%m_%d_%H_%M_%S")
            image_folder = os.path.join(self.base_path, ar_number, "_".join(chunks[1 : ]))

            timesteps = [
                date_time + timedelta(minutes = timestep) 
                for timestep in self.timesteps
            ]
            for img_name in os.listdir(image_folder):
                if img_name.endswith('.jpg'):
                    try:
                        img_name_datetime, img_wavelength = os.path.splitext(img_name)[0].split('__')
                        img_datetime = datetime.strptime(img_name_datetime, "%Y-%m-%dT%H%M%S")
                        datetime_index = [di[0] for di in enumerate(timesteps) if abs(di[1] - img_datetime) < timedelta(minutes=15)]
                        if img_wavelength in self.channels and len(datetime_index) > 0:
                            image_path = os.path.join(image_folder, img_name)
                            image = read_image(image_path,preprocess_image=True)

                            if image.shape != (256, 256, 1):
                                logging.error(f"The shape of image is {image.shape} is not matching expected shape of (256, 256)")
                            
                            if self.data_format == 'channels_last':
                                images[datetime_index[0], :, :, :] = image
                            else:
                                images[:, :, :, datetime_index[0]] = image
                            
                            if isinstance(config["prediction_target"], list):
                                label = [row[col] for col in config["prediction_target"]]
                            elif isinstance(config["prediction_target"], str):
                                label = row[config["prediction_target"]]
                            else:
                                logging.error("The specified prediction target is not supported.")

                            X.append(images)
                            Y.append(label)
                    except Exception as e:
                        logging.error(f"Failed to Load {img_name} due to {e}")
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]

        X = torch.tensor(X, dtype = torch.float)
        Y = torch.tensor(Y, dtype = torch.long)
        return X, Y
