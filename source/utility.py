import pandas as pd
import numpy as np
import json
import runtime_parameters
import torch
import random
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.metrics import fbeta_score
from cv2 import COLOR_BGR2HSV, inRange, getStructuringElement, MORPH_ELLIPSE, erode, bitwise_and, drawContours, THRESH_BINARY_INV, threshold, Canny, dilate, findContours, RETR_TREE, contourArea, CHAIN_APPROX_SIMPLE, cvtColor, COLOR_GRAY2RGB, COLOR_BGR2GRAY, imread, equalizeHist, add, INTER_CUBIC, GaussianBlur, subtract, filter2D, flip, getRotationMatrix2D, warpAffine, imshow, waitKey, destroyAllWindows, resize
from numpy import array, ones, float64, mean, zeros


def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set NumPy random seed
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # Set seed for current GPU
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if you have multiple)
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disable the benchmark to ensure reproducibility

def read_configuration():
    with open(runtime_parameters.CONFIGURATION_FILE_PATH, "r") as fd:
        config = json.load(fd)
    return config

def apply_flare_thresholds(df, flux_column_name) :
    """
    Flux Thresholds has been taken from : https://en.wikipedia.org/wiki/Solar_flare
    """
    df["flare_class"] = np.nan
    df["flare_class"] = np.where(
        df[flux_column_name] < 1e-7,
        "A",
        df["flare_class"]
    )
    df["flare_class"] = np.where(
        (df[flux_column_name] >= 1e-7) &
        (df[flux_column_name] < 1e-6),
        "B",
        df["flare_class"]
    )
    df["flare_class"] = np.where(
        (df[flux_column_name] >= 1e-6) &
        (df[flux_column_name] < 1e-5),
        "C",
        df["flare_class"]
    )
    df["flare_class"] = np.where(
        (df[flux_column_name] >= 1e-5) &
        (df[flux_column_name] < 1e-4),
        "M",
        df["flare_class"]
    )
    df["flare_class"] = np.where(
        df[flux_column_name] >= 1e-4,
        "X",
        df["flare_class"]
    )
    return df

def contrast_stretch(image):
    if len(image.shape) == 3:
        # Split the image into its color channels
        channels = cv2.split(image)
        stretched_channels = []

        for channel in channels:
            min_val = np.min(channel)
            max_val = np.max(channel)
            stretched = (channel - min_val) * (255 / (max_val - min_val))
            stretched = stretched.astype(np.uint8)
            stretched_channels.append(stretched)

        # Merge the stretched channels back together
        stretched_image = cv2.merge(stretched_channels)
        return stretched_image
    else:
        # Find the minimum and maximum pixel values in the image
        min_val = np.min(image)
        max_val = np.max(image)

        # Apply contrast stretching formula
        stretched = (image - min_val) * (255 / (max_val - min_val))
        stretched = stretched.astype(np.uint8)  # Convert back to uint8

        return stretched

def preprocess(
        image,
        resize_shape = None,
        convert2gray = True,
        apply_smoothing = True,
        contrast_streching = True
    ):
    if resize_shape is not None:
        image = cv2.resize(image, resize_shape)

    if convert2gray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if apply_smoothing:
        # Apply Gaussian smoothing
        kernel_size = runtime_parameters.gaussian_kernel_size  # Choose an odd number for the kernel size
        sigma = runtime_parameters.gaussian_sigma  # Let OpenCV calculate sigma
        image = cv2.GaussianBlur(image, kernel_size, sigma)
    
    if contrast_streching:
        image = contrast_stretch(image)

    return image

def read_image(image_path, preprocess_image = True):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    if preprocess_image:
        image = preprocess(
            image,
            resize_shape = runtime_parameters.resize_shape,
            convert2gray = runtime_parameters.convert2gray,
            apply_smoothing = runtime_parameters.apply_smoothing,
            contrast_streching = runtime_parameters.contrast_streching
        )
        # Normalization
        image = image / 255
    image = image.reshape(runtime_parameters.resize_shape[0], runtime_parameters.resize_shape[1], -1)
    return image

def plot_and_save_loss(history, save_dir='plots', filename='loss_plot.png'):
    """
    Plots training and validation loss from a Keras History object and saves the plot.

    Parameters:
    - history: History object returned by model.fit()
    - save_dir: Directory where the plot will be saved (default is 'plots')
    - filename: Name of the file to save the plot as (default is 'loss_plot.png')
    """

    # Plot training and validation loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to: {plot_path}")

def true_skill_score(y_true, y_pred):
    """Calculate True Skill Statistic (TSS) = sensitivity + specificity - 1"""
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    return sensitivity + specificity - 1

def plot_and_save_fbeta_vs_beta(y_true, y_pred_proba, save_dir='plots', filename='loss_plot.png', beta_range=np.linspace(0.1, 2.0, 20), threshold=0.5):
    y_pred_bin = (y_pred_proba >= threshold).astype(int)

    fbeta_scores = []
    for beta in beta_range:
        score = fbeta_score(y_true, y_pred_bin, beta=beta)
        fbeta_scores.append(score)

    plt.plot(beta_range, fbeta_scores, marker='o')
    plt.title('F-beta Score vs Beta')
    plt.xlabel('Beta')
    plt.ylabel('F-beta Score')
    plt.grid(True)
    # Save the plot
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"F-Beta plot saved to: {plot_path}")

def plot_roc_and_pr_curves(y_true, y_pred_proba, save_dir, roc_plot_filename, pr_plot_filename):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    # Save the plot
    plot_path = os.path.join(save_dir, roc_plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC plot saved to: {plot_path}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.title('Precision-Recall (PR) Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    # Save the plot
    plot_path = os.path.join(save_dir, pr_plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"PR plot saved to: {plot_path}")

    return roc_auc, pr_auc



def evaluate_model_performance(dataset, model, model_name):
    y_true_list = []
    y_pred_proba_list = []

    for x_batch, y_batch in dataset:
        y_true_list.append(y_batch.numpy())
        y_pred_proba_list.append(model.predict(x_batch))
    
    y_true = np.concatenate(y_true_list).flatten()
    y_pred_proba = np.concatenate(y_pred_proba_list).flatten()

    best_threshold = 0.5
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    best_f1 = f1_score(y_true, y_pred)
    for threshold in np.linspace(0, 1, 100):
        y_pred = (y_pred_proba >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_threshold = threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)


    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tss = true_skill_score(y_true, y_pred)

    plot_and_save_fbeta_vs_beta(
        y_true, 
        y_pred_proba, 
        save_dir = runtime_parameters.plots_directory, 
        filename = f"{model_name}_FBeta_Plot.png"
    )
    roc_auc, pr_auc = plot_roc_and_pr_curves(
        y_true, 
        y_pred_proba, 
        save_dir = runtime_parameters.plots_directory, 
        roc_plot_filename = f"{model_name}_ROC_Plot.png", 
        pr_plot_filename = f"{model_name}_PR_Plot.png"
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"TSS:       {tss:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")


    return pd.DataFrame({
        "Precision": [precision],
        "Recall":    [recall],
        "F1 Score":  [f1],
        "TSS":       [tss],
        "ROC-AUC" : [roc_auc],
        "PR-AUC" : [pr_auc]
    })

def Rotate(image,rotation=90,steps=15):
    images=[]
    for i in range(-rotation,rotation+1,steps):
        rotation_matrix=getRotationMatrix2D(center=(image.shape[1]/2,image.shape[2]/2),
        angle=i, scale=1)
        rotated_image=warpAffine(src=image, M=rotation_matrix, 
        dsize=(image.shape[1], image.shape[0]))
        images.append(rotated_image)
    return images

def AdjustBrightness(image):
    images=[]
    mask=ones(image.shape,dtype='uint8') * 70
    images.append(add(image,mask))
    images.append(subtract(image,mask))
    return images

def FlipImage(image):
    images=[]
    modes=[-1,0,1]
    for i in modes:
        images.append(flip(image,i))
    return images

def Sharpening(image):
    kernel=array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])  # Laplacian Kernel
    image=filter2D(src=image,ddepth=-1,kernel=kernel)
    return image

def Smoothing(image):
    image=GaussianBlur(image,(3,3),0)
    return image