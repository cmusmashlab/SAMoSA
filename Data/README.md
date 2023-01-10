## 3. Download the SAMoSA dataset
Download the `Data` folder from [here](https://www.dropbox.com/sh/kmv4y3mu7fx8oyp/AAAv1BmCAub00Alp-xPdlA5sa?dl=0) (2.88 GB) and place in `SAMoSA/Data`. 
```bash
Data/
└── TrainingDataset
```

Each file in the `TrainingDataset` folder contains synchronized Audio and IMU data.  
The file name contains all the information regarding the contents of the file.  

For example:  
`49---Kitchen---Chopping---1.pkl`  

Participant ID: 49, Context: Kitchen, Activity: Chopping, TrialNo: 1

Each pickle file has an `Audio` and `IMU` key. Speech (Other) activity files contain an additional `1kHz_Audio` key. 
- `Audio` contains a 1D array of audio data, sampled at 16kHz. We release data for all activities excluding speech at 16kHz. Speech is released at 1kHz (.
- `IMU` is a 2D array of (N\_Samples, 9 axes) sampled at ~50Hz.
- `1kHz_Audio` is a 1D array of audio data, subsampled to 1kHz. Whenever this key is present, the `Audio` key contains an array of zeros with the same shape as the original 16kHz array.  
- Refer to [this notebook](../Code/0.%20Dataset%20Intro.ipynb) and the paper for more information.

