from data_labelling import create_labels, create_binary_target
from data_preprocessing import SDOBenchmarkDataset
import logging
from utility import read_configuration, set_seed, plot_and_save_loss, evaluate_model_performance
from datetime import datetime
import runtime_parameters
from torch.utils.data import DataLoader
import tensorflow as tf
from model import ConvLSTMModel
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

config = read_configuration()
logging.basicConfig(
    filename = config["log_filename"].replace("$$timestamp$$", datetime.now().strftime("%Y_%m_%d_%H_%M_%S")).replace("$$module_name$$", "Orchestrator"),
    level = logging.INFO,
    format='[%(asctime)s] - %(levelname)s -> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == "__main__":
    set_seed(10)
    train_val_df, test_df = create_labels()
    train_val_df = create_binary_target(train_val_df)
    test_df = create_binary_target(test_df)

    # Lets Divide the data into Training, Testing and Validation
    if isinstance(config["prediction_target"], list):
        X = train_val_df.drop(columns = config["prediction_target"])
        Y = train_val_df[config["prediction_target"]]
    else:
        X = train_val_df.drop(columns = [config["prediction_target"]])
        Y = train_val_df[config["prediction_target"]]
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=runtime_parameters.validation_ratio, stratify = Y,random_state=42)

    del X
    del Y
    del train_val_df

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    del X_train
    del y_train
    del X_val
    del y_val

    # Create the Parsed-structured Dataset objects for training, validation and testing model performance.
    train_dataset = SDOBenchmarkDataset(config["dataset_path"]["training_data"],train_df, data_format="channels_last")
    val_dataset = SDOBenchmarkDataset(config["dataset_path"]["training_data"],val_df, data_format="channels_last")

    del train_df
    del val_df
    # train_dataloader = DataLoader(train_dataset, batch_size=runtime_parameters.batch_size, shuffle = True)
    # val_dataloader = DataLoader(val_dataset, batch_size=runtime_parameters.batch_size, shuffle = False)
    # test_dataloader = DataLoader(test_dataset, batch_size=runtime_parameters.batch_size, shuffle = False)


    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset.X, train_dataset.Y))
    tf_train_dataset = tf_train_dataset.shuffle(buffer_size=1000,reshuffle_each_iteration=True).batch(runtime_parameters.batch_size).prefetch(tf.data.AUTOTUNE)

    tf_val_dataset = tf.data.Dataset.from_tensor_slices((val_dataset.X, val_dataset.Y))
    tf_val_dataset = tf_val_dataset.batch(runtime_parameters.batch_size).prefetch(tf.data.AUTOTUNE)

    # Model Instantiation
    n_frames, height, width, in_channels = runtime_parameters.n_stacked_frames, runtime_parameters.resize_shape[0], runtime_parameters.resize_shape[1], runtime_parameters.image_channels

    convlstm_model = ConvLSTMModel(
        n_frames = n_frames,
        height = height,
        width = width,
        in_channels = in_channels,
        n_stacked_convlstm_layers = runtime_parameters.n_stacked_convlstm_layers
    )

    # Build the model
    convlstm_model.build(input_shape=(None, n_frames, height, width, in_channels))
    x = np.random.rand(32, n_frames, height, width, in_channels)
    convlstm_model(x)
    print(convlstm_model.summary())

    del x

    # Compile the model
    convlstm_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam() 
    )

    # Lets define some callbacks to Adjust LR and Early Stopping.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=runtime_parameters.early_stopping_patience)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=runtime_parameters.lr_scheduler_patience)

    history = convlstm_model.fit(
        tf_train_dataset,
        batch_size = runtime_parameters.batch_size,
        verbose = runtime_parameters.verbose,
        callbacks = [
            early_stopping,
            reduce_lr
        ],
        validation_data = tf_val_dataset
    )

    del train_dataset
    del tf_train_dataset

    # Evaluate the model
    test_dataset = SDOBenchmarkDataset(config["dataset_path"]["test_data"],test_df, data_format="channels_last")
    tf_test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset.X, test_dataset.Y))
    tf_test_dataset = tf_test_dataset.batch(runtime_parameters.batch_size).prefetch(tf.data.AUTOTUNE)

    ## Create the saving directory
    ### Create directory if it doesn't exist
    os.makedirs(runtime_parameters.outputs_directory, exist_ok=True)
    os.makedirs(runtime_parameters.plots_directory, exist_ok=True)

    ## Start with generating the plot of loss history
    plot_and_save_loss(
        history, 
        save_dir = runtime_parameters.plots_directory, 
        filename = "ConvLSTM_loss_history.png"
    )
    ## Evaluate the performance of model on below metrics
    ###   - Precision
    ###   - Recall
    ###   - F1 Score
    ###   - True Skill Score
    ###   - Fbeta Plot
    ###   - ROC-AUC
    ###   - ROC Plot
    ###   - PR-AUC
    ###   - PR Plot
    val_metrics = evaluate_model_performance(tf_val_dataset, convlstm_model)
    test_metrics = evaluate_model_performance(tf_test_dataset, convlstm_model)
    
    ## Save evaluation metrics
    val_metrics = val_metrics.T
    val_metrics.columns = ["Metrics"]
    test_metrics = test_metrics.T
    test_metrics.columns = ["Metrics"]

    val_metrics.to_excel(os.path.join(runtime_parameters.outputs_directory, "validation_metrics.xlsx"))
    test_metrics.to_excel(os.path.join(runtime_parameters.outputs_directory, "test_metrics.xlsx"))





