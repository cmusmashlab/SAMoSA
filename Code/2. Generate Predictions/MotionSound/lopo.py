import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # select gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# imports go here
import gc
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import balanced_accuracy_score, f1_score

from collections import defaultdict
import datetime

import params

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

def get_normalization_params(data_path, train_pids):
    X_imu = []

    file_path_gen = data_path.iterdir
    for pkl_file in tqdm(file_path_gen(), total=len([*file_path_gen()])):
        pid, context, activity, trial_no = pkl_file.name.strip("*.pkl").split("---")

        if pid in train_pids:
            with open(pkl_file, "rb") as f:
                file_data = pkl.load(f)["IMU"] # shape = (samples, features(acc_x,y,z, gyr, ori))
                if file_data.shape[0] == 0:
                    continue
            X_imu.append(file_data)

    X_imu_concat = np.concatenate(X_imu, axis=0) # keep only the right imu

    # get 80th percentile
    pseudo_max = np.percentile(X_imu_concat, 80, axis=0, keepdims=True)
    pseudo_min = np.percentile(X_imu_concat, 20, axis=0, keepdims=True)

    # get mean
    mean = np.mean(X_imu_concat, axis=0, keepdims=True)
    std = np.std(X_imu_concat, axis=0, keepdims=True)

    return {"max": pseudo_max, "min": pseudo_min, "mean": mean, "std": std}

def load_data(path_to_data, load_pids_set, normalization_params):
    X_imu = []
    X_audio = []
    y = []

    # unpack normalization params
    pseudo_max = normalization_params["max"]
    pseudo_min = normalization_params["min"]
    mean = normalization_params["mean"]
    std = normalization_params["std"]

    file_path_gen = path_to_data.iterdir
    for file in tqdm(file_path_gen(), total=len([*file_path_gen()])):
        pid, context, activity, trial_no = file.name.strip(".pkl").split("---")

        if pid in load_pids_set:
            with open(file, "rb") as f:
                file_dataset = pkl.load(f)
                file_data = file_dataset["IMU"]
                audio_examples = file_dataset["audio"]
                if file_data.shape[0] == 0:
                    continue
                
            # normalize data
            file_data_normalized = 1 + (file_data - pseudo_max)*(2)/(pseudo_max - pseudo_min)

            # standardize data to mean 0 and std 1
            file_data_normalized = (file_data_normalized - mean) / std
            imu_examples = file_data_normalized

            X_imu.append(imu_examples)
            X_audio.append(audio_examples)
            for _ in range(len(imu_examples)):
                y.append([pid, context, activity, trial_no])

    X_imu_concat = np.concatenate(X_imu, axis=0) 
    X_audio_concat = np.concatenate(X_audio, axis=0)

    y_df = pd.DataFrame(y, columns=["PID", "Context", "Activity", "Trial_No"])

    return X_imu_concat, X_audio_concat, y_df

# callbacks
def get_mean_filewise_acc(loc_model, test_pids, labelBinarizer=None, 
                          path_to_data=None, return_df=False,
                          experiment="both", normalization_params=None):    
    # predict file wise         
    accuracy_dict = defaultdict(list)     

    X_imu = []
    X_audio = []
    y = []


    # unpack normalization params
    pseudo_max = normalization_params["max"]
    pseudo_min = normalization_params["min"]
    mean = normalization_params["mean"]
    std = normalization_params["std"]

    file_path_gen = path_to_data.iterdir
    for file in tqdm(file_path_gen(), total=len([*file_path_gen()])):
        pid, context, activity, trial_no = file.name.strip(".pkl").split("---")

        if pid in test_pids:
            with open(file, "rb") as f:
                file_dataset = pkl.load(f)
                file_data = file_dataset["IMU"]
                audio_examples = file_dataset["audio"]
                if file_data.shape[0] == 0:
                    continue
                
            # normalize data
            file_data_normalized = 1 + (file_data - pseudo_max)*(2)/(pseudo_max - pseudo_min)

            # standardize data to mean 0 and std 1
            file_data_normalized = (file_data_normalized - mean) / std
            imu_examples = file_data_normalized
            
            X_imu.append(imu_examples)
            X_audio.append(audio_examples)
            for _ in range(len(imu_examples)):
                y.append([file.name])

    imu_inputs = np.concatenate(X_imu, axis=0) 
    audio_inputs = np.concatenate(X_audio, axis=0)
    y_true = pd.DataFrame(y, columns=["file_name"])    
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

# set random vars
import random
random.seed(4)
np.random.seed(4)
tf.random.set_seed(4)

# set up stuff to save
path_to_preds = Path("../../../Preds/MotionSound")
path_to_preds.mkdir(exist_ok=True, parents=True)

path_to_models = Path("../../../Models/MotionSound/")

# +
path_to_examples = Path("../../../Data/MMExamples/")

sub_srs = []

for folder in path_to_examples.iterdir():
    sub_srs.append(folder.name)

print(sub_srs)
# -

for sr in sub_srs:
    sub_sr = int(sr)

    # Prepare data
    path_to_mm_data = path_to_examples / sr

    # imu params
    imu_sr = 50
    window_len_imu = 2 * imu_sr  # 2 secs of data worth of samples
    hop_len_imu = 10

    # get train test split
    pids_set = get_pids_set(path_to_mm_data)
    print(pids_set)
    print(sr)

    # start lopo stuff
    for _ in pids_set:
        train_pids_set, val_pids_set, test_pids_set = get_train_val_test_pids(pids_set, split=[18, 1, 1])
        
        model_name = '_'.join(test_pids_set)
        path_to_model = path_to_models / sr / f"{model_name}/{model_name}.h5"
        multimodal_model = tf.keras.models.load_model(path_to_model)

        with open(path_to_model.parent / "norm_params.pkl", "rb") as f:
            normalization_params = pkl.load(f)
        X_train_imu, X_train_audio, y_train = load_data(path_to_mm_data, train_pids_set, normalization_params=normalization_params)
        X_val_imu, X_val_audio, y_val = load_data(path_to_mm_data, val_pids_set, normalization_params=normalization_params)

        print(X_train_imu.shape, X_train_audio.shape, y_train.shape)
        print(X_val_imu.shape, X_val_audio.shape, y_val.shape)

        y_train_df = y_train
        y_val_df = y_val

        Y_train_activity = y_train_df["Activity"]
        Y_val_activity = y_val_df["Activity"]

        lb = LabelBinarizer()
        Y_train_activity_lbl = lb.fit_transform(Y_train_activity)
        Y_val_activity_lbl = lb.transform(Y_val_activity)

        results_df = get_mean_filewise_acc(multimodal_model, test_pids_set, path_to_data=path_to_mm_data,
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
        path_to_save_preds = path_to_preds / f"{sub_sr}"
        path_to_save_preds.mkdir(exist_ok=True, parents=True)
        results_df.to_csv(path_to_save_preds/f'{"_".join(test_pids_set)}.csv', index=False)

        # clear session
        K.clear_session()

        # delete all the unused vars
        del multimodal_model
        del X_train_audio, X_train_imu, y_train_df
        del X_val_audio, X_val_imu, y_val_df
        gc.collect()

        # end
        pids_set = np.roll(pids_set, 1)  # roll the array for cross validation



