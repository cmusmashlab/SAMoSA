## 4. Download the SAMoSA models
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

