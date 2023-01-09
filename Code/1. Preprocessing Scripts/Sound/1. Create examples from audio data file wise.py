# %%
import pickle as pkl
import numpy as np
from pathlib import Path
import vggish_input

# %%
path_to_final = Path("../../../Data/TrainingDataset/")

# %%
path_to_save = Path("../../../Data/AudioExamples")
path_to_save.mkdir(exist_ok=True, parents=True)

# %%
# sub_srs = [16000, 8000, 4000, 2000, 1000, 500, 250, 125] # choose your sub sampling rates
sub_srs = [16000, 1000]

# %%
for file_path in (path_to_final).iterdir():
    print(f"----{file_path.name}----")
    example_shapes = []
    with open(file_path, "rb") as f:
        fdata = pkl.load(f)
    
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
        audio_examples = vggish_input.wavform_to_examples(wav_data, lower_edge_hertz=leh, upper_edge_hertz=ueh, sr=sub_sr)
        example_shapes.append(audio_examples.shape)
        # save examples as pkl
        
        with open(path_to_save_sub_sr / f"{file_path.name}", "wb") as f:
            pkl.dump(audio_examples, f)
            
    if not(example_shapes.count(example_shapes[0]) == len(example_shapes)):
        print(example_shapes)
