# SAMoSA: Sensing Activities with Motion and Subsampled Audio

Research code for SAMoSA: Sensing Activities with Motion and Subsampled Audio (Ubicomp/IMWUT 2022).

## Reference
Vimal Mollyn, Karan Ahuja, Dhruv Verma, Chris Harrison, and Mayank Goel. 2022. SAMoSA: Sensing Activities with Motion and Subsampled Audio. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 6, 3, Article 132 (September 2022), 19 pages. 

[Download Paper Here.](http://smashlab.io/pdfs/samosa.pdf) 

BibTeX Reference: 
```
@article{samosa,
    author = {Mollyn, Vimal and Ahuja, Karan and Verma, Dhruv and Harrison, Chris and Goel, Mayank},
    title = {SAMoSA: Sensing Activities with Motion and Subsampled Audio},
    year = {2022},
    issue_date = {September 2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {6},
    number = {3},
    url = {https://doi.org/10.1145/3550284},
    doi = {10.1145/3550284},
    journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
    month = {sep},
    articleno = {132},
    numpages = {19},
    keywords = {Sensors, Location-Aware/Contextual Computing, Artifact or System}
}
```

## 1. Clone this repository
```
git clone https://github.com/cmusmashlab/SAMoSA.git
```

## 2. Create a virtual environment
We recommend using conda. Tested on `Ubuntu 20.04`, with `python 3.8`.

```
conda create -n "samosa" python=3.8.3
conda activate samosa
conda install -c conda-forge cudatoolkit=10.2 cudnn=7.6

python -m pip install -r requirements.txt
```

## 3. Download Data
Download the `Data` folder from [UPDATE THIS LINK!](http://smashlab.io/pdfs/samosa.pdf) and place in `SAMoSA/Data`. It should have the following tree structure.
```bash
Data/
└── TrainingDataset
```

Each file in the `TrainingDataset` folder contains synchronized Audio and IMU data.  
The file name contains all the information regarding the contents of the file.  

For example:  
`49---Kitchen---Chopping---1.pkl`  

Participant ID: 49, Context: Kitchen, Activity: Chopping, TrialNo: 1

Each pickle file has the `Audio` and `IMU` keys.  
    - `Audio` contains a 1D array of audio data, sampled at 16kHz. We have released all activities excluding speech at 16kHz. Speech is released at 1kHz.
    - `IMU` is a 2D array of (N\_Samples, 9 axes) sampled at ~50Hz
    - Refer to [this notebook](Code/0.%20Dataset%20Intro.ipynb) and the paper for more information.

## 4. Download Models
Download the `Models` folder from [UPDATE THIS LINK!](http://smashlab.io/pdfs/samosa.pdf) and place in `SAMoSA/Models`. It should have the following tree structure.
```bash
Models/
├── Motion
├── MotionSound
│   ├── 1000
│   └── 16000
└── Sound
    ├── 1000
    └── 16000
```

## 5. Generate Predictions for each model type
Run scripts in `Code/1. Generate Predictions`. This will generate predictions in a new folder located at `SAMoSA/Preds`. This will generate the following.
```bash
Preds/
├── Motion
├── MotionSound
│   ├── 1000
│   └── 16000
└── Sound
    ├── 1000
    └── 16000
```
