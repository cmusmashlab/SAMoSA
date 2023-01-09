# imports go here
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # select gpu
import gc
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, f1_score

from collections import defaultdict

# set random vars
import random
random.seed(4)
np.random.seed(4)
tf.random.set_seed(4)

# path to save predictions
path_to_save_preds = Path("../../../Preds/Motion/")
path_to_save_preds.mkdir(exist_ok=True, parents=True)

# Prepare data and models
path_to_raw_imu = Path("../../../Data/TrainingDataset/")
path_to_motion_models = Path("../../../Models/Motion/")

# imu params
imu_sr = 50
window_len_imu = 2 * imu_sr  # 2 secs of data
hop_len_imu = 10


def frame(data, window_length, hop_length):
    # pad zeros if sequence too short
    if data.shape[0] < window_length:
        len_pad = int(np.ceil(window_length)) - data.shape[0]
        to_pad = np.zeros((len_pad, ) + data.shape[1:])
        data = np.concatenate([data, to_pad], axis=0)
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, int(window_length)) + data.shape[1:]
    strides = (data.strides[0] * int(hop_length),) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def get_pids_set(data_path):
    # create train, val, test splits
    file_path_gen = data_path.iterdir
    pids = []
    for pkl_file in file_path_gen():
        pid, context, activity, trial_no = pkl_file.name.strip("*.pkl").split("---")
        pids.append(pid)
    pids_set = sorted(list(set(pids)))
    # get train, test, val split for pids
    np.random.seed(4)
    np.random.shuffle(pids_set)
    return pids_set

def get_train_val_test_pids(pids_set, split=[9, 2, 1]):
    train_pids_set = pids_set[:split[0]]
    val_pids_set = pids_set[split[0]:split[0] + split[1]]
    test_pids_set = pids_set[split[0] + split[1]:]
    print(f"Train participants: {train_pids_set}")
    print(f"Val participants: {val_pids_set}")
    print(f"Test participants: {test_pids_set}")
    return train_pids_set, val_pids_set, test_pids_set

def load_data(data_path, pids_set, normalization_params):
    X_imu = []
    y = []

    # unpack normalization params
    pseudo_max = normalization_params["max"]
    pseudo_min = normalization_params["min"]
    mean = normalization_params["mean"]
    std = normalization_params["std"]
    
    file_path_gen = data_path.iterdir
    for pkl_file in tqdm(file_path_gen(), total=len([*file_path_gen()])):
        pid, context, activity, trial_no = pkl_file.name.strip("*.pkl").split("---")
        
        if pid in pids_set:
            with open(pkl_file, "rb") as f:
                file_data = pkl.load(f)["IMU"] # shape = (samples, features(acc_x,y,z, gyr, ori))

            # normalize data to ~[-1, 1]
            file_data_normalized = 1 + (file_data - pseudo_max)*(2)/(pseudo_max - pseudo_min)
            
            # standardize data to mean 0 and std 1
            file_data_normalized = (file_data_normalized - mean) / std
            
            # window data
            windowed_data = frame(file_data_normalized, window_len_imu, hop_len_imu)
        
            X_imu.append(windowed_data)
            for _ in range(len(windowed_data)):
                y.append([pid, context, activity, trial_no])

    X_imu_concat = np.concatenate(X_imu, axis=0) 

    y_df = pd.DataFrame(y, columns=["PID", "Context", "Activity", "Trial_No"])

    return X_imu_concat, y_df

def get_normalization_params(data_path, train_pids_set):
    X_imu = []
    
    file_path_gen = data_path.iterdir
    for pkl_file in tqdm(file_path_gen(), total=len([*file_path_gen()])):
        pid, context, activity, trial_no = pkl_file.name.strip("*.pkl").split("---")
        
        if pid in pids_set:
            with open(pkl_file, "rb") as f:
                file_data = pkl.load(f)["IMU"] # shape = (samples, features(acc_x,y,z, gyr, ori))
            X_imu.append(file_data)
            
    X_imu_concat = np.concatenate(X_imu, axis=0)
    
    # path to save predictions
    # get 80th percentile
    pseudo_max = np.percentile(X_imu_concat, 80, axis=0, keepdims=True)
    pseudo_min = np.percentile(X_imu_concat, 20, axis=0, keepdims=True)
    
    # get mean
    mean = np.mean(X_imu_concat, axis=0, keepdims=True)
    std = np.std(X_imu_concat, axis=0, keepdims=True)
    
    # get std
    
    return {"max": pseudo_max, "min": pseudo_min, "mean": mean, "std": std}

# get train test split
pids_set = get_pids_set(path_to_raw_imu)

# callbacks
def get_mean_filewise_acc(loc_model, test_pids, labelBinarizer=None,                    
        path_to_preprocessed_data=None, return_df=False, experiment="imu", normalization_params=None):    
    # predict file wise         
    accuracy_dict = defaultdict(list)     
    
    file_path_gen = path_to_preprocessed_data.iterdir    
                         
    imu_inputs = []                
    y_true = []    
                                                           
    # unpack normalization params
    pseudo_max = normalization_params["max"]
    pseudo_min = normalization_params["min"]
    mean = normalization_params["mean"]
    std = normalization_params["std"]
    
    for pkl_file in file_path_gen():                       
        pid, context, activity, trial_no = pkl_file.name.strip(".wav").split("---")    
        if pid in test_pids:                          
            # load data from preprocessed path    
            with open(pkl_file, "rb") as f:
                file_data = pkl.load(f)["IMU"] # shape = (samples, features(acc_x,y,z, gyr, ori))

            # normalize data to ~[-1, 1]
            file_data_normalized = 1 + (file_data - pseudo_max)*(2)/(pseudo_max - pseudo_min)
            
            # standardize data to mean 0 and std 1
            file_data_normalized = (file_data_normalized - mean) / std
            
            # window data
            windowed_data = frame(file_data_normalized, window_len_imu, hop_len_imu)
            # print(windowed_data.shape)

            if pid in pids_set:
                imu_inputs.append(windowed_data)
                for _ in range(len(windowed_data)):
                    y_true.append([pkl_file.name])
            
    imu_inputs = np.concatenate(imu_inputs, axis=0)
    y_true = pd.DataFrame(y_true, columns=["file_name"])    
    y_pred_df = []
    
        
    # pass through model    
    if experiment=="both":    
        preds = loc_model.predict([imu_inputs, audio_inputs], batch_size=1024)    
    elif experiment=="audio":    
        preds = loc_model.predict(audio_inputs, batch_size=1024)    
    elif experiment=="imu":    
        preds = loc_model.predict(imu_inputs, batch_size=1024)   
        
    results_df = pd.DataFrame(preds, columns=lb.classes_)
    results_df["file_name"] = y_true
    return results_df

for _ in pids_set:
    train_pids_set, val_pids_set, test_pids_set = get_train_val_test_pids(pids_set, split=[18, 1, 1])

    normalization_params = get_normalization_params(path_to_raw_imu, train_pids_set)
    X_train_imu, y_train = load_data(path_to_raw_imu, train_pids_set, normalization_params)
    X_val_imu, y_val = load_data(path_to_raw_imu, val_pids_set, normalization_params)

    y_train_df = y_train
    y_val_df = y_val

    Y_train_activity = y_train_df["Activity"]
    Y_val_activity = y_val_df["Activity"]

    lb = LabelBinarizer()
    Y_train_activity_lbl = lb.fit_transform(Y_train_activity)
    Y_val_activity_lbl = lb.transform(Y_val_activity)

    train_class_weights = compute_class_weight("balanced", classes=sorted(np.unique(Y_train_activity)),
                                               y=Y_train_activity)

    # get train sample weights
    train_sample_weights = np.zeros((len(Y_train_activity)))
    classes = sorted(np.unique(Y_train_activity))

    for i, class_label in enumerate(classes):
        train_sample_weights[Y_train_activity == class_label] = train_class_weights[i]

    # load the model
    path_to_model = path_to_motion_models / f'{"_".join(test_pids_set)}.h5'
    
    model = tf.keras.models.load_model(path_to_model)

    # save results to file
    results_df = get_mean_filewise_acc(model, test_pids_set, path_to_preprocessed_data=path_to_raw_imu,
                                       labelBinarizer=lb, normalization_params=normalization_params, return_df=True)
    
    results_df["y_pred"] = results_df.drop(columns=["file_name"]).idxmax(axis=1)
    results_df["y_true"] = results_df["file_name"].str.split("---").str[2]
    ba_framewise = balanced_accuracy_score(results_df["y_true"], results_df["y_pred"])
    f1_framewise = f1_score(results_df["y_true"], results_df["y_pred"], average="weighted")
    
    file_preds = []
    for name, group in results_df.groupby(["file_name"]):
        preds = group.drop(columns = ["file_name", "y_pred", "y_true"])
        file_pred = preds.sum(axis=0).idxmax(axis=0)
        file_true = group["y_true"].values[0]

        file_preds.append([file_true, file_pred])
    file_preds = pd.DataFrame(file_preds, columns=["file_true", "file_pred"])

    ba_filewise = balanced_accuracy_score(file_preds["file_true"], file_preds["file_pred"])
    f1_filewise = f1_score(file_preds["file_true"], file_preds["file_pred"], average="weighted")
    
    print(f'{"_".join(test_pids_set)}, {"_".join(val_pids_set)}, {ba_framewise}, {f1_framewise}, {ba_filewise}, {f1_filewise}')
        
    # save predictions
    results_df.to_csv(path_to_save_preds/f'{"_".join(test_pids_set)}.csv', index=False)

    # clear session
    K.clear_session()

    # end
    pids_set = np.roll(pids_set, 1)  # roll the array for cross validation

    # delete all the unused vars
    del model
    del X_train_imu, y_train_df
    del X_val_imu, y_val_df
    gc.collect()


