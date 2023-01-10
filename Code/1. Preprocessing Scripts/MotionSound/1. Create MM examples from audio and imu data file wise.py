# %%
import pickle as pkl
import numpy as np
from pathlib import Path
import vggish_input
import params
from tqdm.notebook import tqdm_notebook as tqdm

# %%
path_to_final = Path("../../../Data/TrainingDataset/")

# %%
path_to_save = Path("../../../Data/MMExamples")
path_to_save.mkdir(exist_ok=True, parents=True)

# %%
# sub_srs = [16000, 8000, 4000, 2000, 1000, 500, 250, 125] # select sub sampling rates
sub_srs = [16000, 1000]

# %%
# imu params
imu_sr = 50
window_len_imu = 2 * imu_sr  # 2 secs of data worth of samples
hop_len_imu = 10


# %%
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


# %%
for file_path in (path_to_final).iterdir():
    print(f"----{file_path.name}----")
    example_shapes = []
    
    # load data
    with open(file_path, "rb") as f:
        fdata = pkl.load(f)    
        
    # window data
    imu_examples = frame(fdata["IMU"], window_len_imu, hop_len_imu)
    
    og_audio = fdata["Audio"]
    for sub_sr in sub_srs:
        path_to_save_sub_sr = path_to_save / str(sub_sr)
        path_to_save_sub_sr.mkdir(exist_ok=True, parents=True)

        
        # subsample the audio
        factor = 16000 // sub_sr

        wav_data = og_audio[::factor].astype(np.int16)
        
        # create examples from wavfile
        leh = 10
        ueh = sub_sr//2
        audio_examples = vggish_input.wavform_to_concat_examples(wav_data,
                                                                 lower_edge_hertz=leh,
                                                                 upper_edge_hertz=ueh, sr=sub_sr) # shape = samples, num_mel bins
        
        # for each example in imu examples, find the most recent audio example based on timestamp
        audio_timestamps = []
        # get timestamps for each index for audio
        for i in range(audio_examples.shape[0]):
            timestamp = i*params.STFT_HOP_LENGTH_SECONDS + params.STFT_WINDOW_LENGTH_SECONDS
            audio_timestamps.append(timestamp)

        windowed_data_audio = []
        windowed_data_imu = []
        
        for i in range(imu_examples.shape[0]):
            # get end time 
            end_sample_imu = i*hop_len_imu + window_len_imu # units samples
            end_time_imu = end_sample_imu/50
            hop_len_secs_imu = hop_len_imu/50
            window_len_secs_imu = window_len_imu/50

            j = int((i*hop_len_secs_imu + window_len_secs_imu - params.STFT_WINDOW_LENGTH_SECONDS)/params.STFT_HOP_LENGTH_SECONDS)
            
            end_index = j
            example_window_samples = int(params.EXAMPLE_WINDOW_SECONDS/(params.STFT_HOP_LENGTH_SECONDS))
            start_index = end_index - example_window_samples
            if start_index < 0:
                # drop this imu_example as well
                continue
            else:
                audio_example = audio_examples[start_index: end_index]
                # pad
                to_pad = example_window_samples - audio_example.shape[0]
                zero_pad = np.zeros((to_pad, ) + audio_example.shape[1:])
                audio_example = np.concatenate([audio_example, zero_pad], axis=0)
            windowed_data_audio.append(audio_example)
            windowed_data_imu.append(imu_examples[i])
            
        windowed_arr_audio = np.array(windowed_data_audio)
        windowed_arr_imu = np.array(windowed_data_imu)
        
        assert windowed_arr_imu.shape[0] == windowed_arr_audio.shape[0], f"Something's still wrong!, {file.name, audio_examples.shape, imu_examples.shape}"
        example_shapes.append(windowed_arr_audio.shape)
        
        # create dataset
        dataset = {
            "IMU": windowed_arr_imu,
            "audio": windowed_arr_audio
        }
        
        # save examples as pkl
        with open(path_to_save_sub_sr / f"{file_path.name[:-4]}.pkl", "wb") as f:
            pkl.dump(dataset, f)
            
    if not(example_shapes.count(example_shapes[0]) == len(example_shapes)):
        print(example_shapes)
