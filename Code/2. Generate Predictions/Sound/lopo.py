import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import balanced_accuracy_score, f1_score
from pathlib import Path

import random

from tqdm import trange

random.seed(4)
np.random.seed(4)
tf.random.set_seed(4)

from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import pickle as pkl

# +
path_to_examples = Path("../../../Data/AudioExamples/")

sub_srs = []

for folder in path_to_examples.iterdir():
    sub_srs.append(folder.name)

print(sub_srs)


# -

def load_data(path_to_data, load_pids_set):
    X_audio = []
    y = []

    file_path_gen = path_to_data.iterdir
    for file_path in tqdm(file_path_gen(), total=len([*file_path_gen()])):
        pid, context, activity, trial_no = file_path.name.strip(".pkl").split("---")

        if pid in load_pids_set:
            with open(file_path, "rb") as f:
                audio_examples = pkl.load(f) # shape = samples, num_mel bins

            X_audio.append(audio_examples)
            for _ in range(len(audio_examples)):
                y.append([pid, context, activity, trial_no])
                
    X_audio_concat = np.concatenate(X_audio, axis=0)
    y_df = pd.DataFrame(y, columns=["PID", "Context", "Activity", "Trial_No"])
    return X_audio_concat, y_df

path_to_sub_sr_examples = sub_srs[0]

def get_mean_filewise_acc(loc_model, test_pids, labelBinarizer=None,
                          path_to_data=None, return_df=False, 
                          experiment="audio"):
    # predict file wise
    accuracy_dict = defaultdict(list)
    X_audio = []
    y = []

    file_path_gen = path_to_data.iterdir
    for file_path in tqdm(file_path_gen(), total=len([*file_path_gen()])):
        pid, context, activity, trial_no = file_path.name.strip(".pkl").split("---")

        if pid in test_pids:
            with open(file_path, "rb") as f:
                audio_examples = pkl.load(f)

            X_audio.append(audio_examples)
            for _ in range(len(audio_examples)):
                y.append([file_path.name])
                
    audio_inputs = np.concatenate(X_audio, axis=0)
    y_true = pd.DataFrame(y, columns=["file_name"])
    y_pred_df = []
    
    # pass through model
    if experiment == "both":
        preds = loc_model.predict([imu_inputs, audio_inputs], batch_size=1024)
    elif experiment == "audio":
        preds = loc_model.predict(audio_inputs, batch_size=1024)
    elif experiment == "imu":
        preds = loc_model.predict(imu_inputs, batch_size=1024)
        
    results_df = pd.DataFrame(preds, columns=lb.classes_)
    results_df["file_name"] = y_true
    return results_df

def get_pids_set(data_path):
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

# set up some paths and filenames
path_to_preds = Path("../../../Preds/Sound")
path_to_preds.mkdir(exist_ok=True, parents=True)

path_to_models = Path("../../../Models/Sound/")

# load data
# get set of factors
for sr in sub_srs:
    random.seed(4)
    np.random.seed(4)
    tf.random.set_seed(4)

    sub_sampling_rate = int(sr)
    path_to_data = path_to_examples / sr
    pids_set = get_pids_set(path_to_data)
    print(pids_set)
    print(sr)

    for _ in trange(len(pids_set)): # lopo for all pids
        train_pids_set, val_pids_set, test_pids_set = get_train_val_test_pids(pids_set, split=[18, 1, 1])

        path_to_model = path_to_models / sr / f"{'_'.join(test_pids_set)}.h5"
        if path_to_model.exists():
            final_model = tf.keras.models.load_model(path_to_model)
            
            # load data
            X_train_audio_concat, y_train_df = load_data(path_to_data, train_pids_set)
            X_val_audio_concat, y_val_df = load_data(path_to_data, val_pids_set)

            Y_train_activity = y_train_df["Activity"]
            Y_val_activity = y_val_df["Activity"]

            lb = LabelBinarizer()
            Y_train_activity_lbl = lb.fit_transform(Y_train_activity)
            Y_val_activity_lbl = lb.transform(Y_val_activity)

            # save results to file
            results_df = get_mean_filewise_acc(final_model, test_pids_set, path_to_data=path_to_data,
                                               labelBinarizer=lb, return_df=True)

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
            path_to_save_preds = path_to_preds / f"{sub_sampling_rate}"
            path_to_save_preds.mkdir(exist_ok=True, parents=True)
            results_df.to_csv(path_to_save_preds/f'{"_".join(test_pids_set)}.csv', index=False)

            K.clear_session()
            del final_model
            del X_train_audio_concat, y_train_df
            del X_val_audio_concat, y_val_df
            gc.collect()

        # end
        pids_set = np.roll(pids_set, 1)  # roll the array for cross validation



