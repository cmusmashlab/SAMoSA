# SAMoSA: Sensing Activities with Motion and Subsampled Audio

Research code for SAMoSA: Sensing Activities with Motion and Subsampled Audio (Ubicomp/IMWUT 2022).

![](Media/SAMoSA_a.gif)
![](Media/SAMoSA_b.gif)

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
- Refer to [this notebook](Code/0.%20Dataset%20Intro.ipynb) and the paper for more information.

## 4. Download Models
Download the `Models` folder from [here](https://www.dropbox.com/sh/ly2k47llynd4bl2/AACIqr9BCzmxsBbwS0xVML59a?dl=0) (2.46 GB) and place in `SAMoSA/Models`. It should have the following tree structure. Note, to replicate results from the paper, you must download all 20 models from [here](https://www.dropbox.com/sh/hc302qcid5nnkes/AADtsM5KeNTDrEX5TjLwA54oa?dl=0) (49.5 GB).
```bash
Models/
├── Motion
│   └── 17.h5
├── MotionSound
│   ├── 1000
│   │   └── 17
│   │       ├── 17.h5
│   │       ├── lb.pkl
│   │       └── norm_params.pkl
│   └── 16000
│       └── 17
│           ├── 17.h5
│           ├── lb.pkl
│           └── norm_params.pkl
└── Sound
    ├── 1000
    │   └── 17.h5
    └── 16000
        └── 17.h5
```

## 5. Generate Predictions for each model type
Run scripts in `Code/1. Generate Predictions`. This will generate predictions in a new folder located at `SAMoSA/Preds`, with the following structure.
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

## 6. Run Evaluation notebooks 
Note, results from the paper are averaged across all 20 Leave-One-Participant-Out (LOPO) cross validation folds.
- [Evaluation notebook for context wise metrics](Code/3.%20Evaluate/1.%20Context%20Wise%20results.ipynb)
- [Evaluation notebook for frame wise metrics](Code/3.%20Evaluate/2.%20Frame%20Wise%20metrics.ipynb)

## Annotator Web App
The annotator app that we used during data collection can be found [here](https://github.com/VimalMollyn/UserStudyAnnotator).

## Coming Soon!
- Real-Time demos and visualizations
- Android WearOS App for data collection and streaming

## Disclaimer
```
THE PROGRAM IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW THE AUTHOR WILL BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```

## Contact
Feel free to contact [Vimal Mollyn](mailto:ms123vimal@gmail.com) for any help, questions or general feedback!
