# SAMoSA: Sensing Activities with Motion and Subsampled Audio

Research code for SAMoSA: Sensing Activities with Motion and Subsampled Audio (Ubicomp/IMWUT 2022).

## Reference
Vimal Mollyn, Karan Ahuja, Dhruv Verma, Chris Harrison, and Mayank Goel. 2022. SAMoSA: Sensing Activities with Motion and Subsampled Audio. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 6, 3, Article 132 (September 2022), 19 pages. https://doi.org/10.1145/3550284

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

