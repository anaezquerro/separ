# :seedling: [Se]()quence Labeling [Par]()sing

Hi :wave: This is a [Python](https://www.python.org/) implementation for sequence labeling algorithms (and others!) for Dependency, Constituency and Semantic Parsing.


- **Dependency Parsing**:
    - Absolute and relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)).
    - PoS-tag relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)).
    - Bracketing encoding ($k$-planar) ([Strzyz et al., 2020](https://aclanthology.org/2020.coling-main.223/)).
    - Arc-Eager transition-based system ([Nivre & Fernández-González, 2002](https://aclanthology.org/J14-2002/)).
    - $4$-bit projective  encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)).
    - $7$-bit $2$-planar encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)).
- **Semantic Dependency Parsing**:
    - Absolute and relative indexing ([Ezquerro et al., 2024]()).
    - Bracketing encoding ($k$-planar) ([Ezquerro et al., 2024]()).
    - $4k$-bit encoding ($k$-planar) ([Ezquerro et al., 2024]()).
    - $6k$-bit encoding ($k$-planar) ([Ezquerro et al., 2024]()).
- **Constituency Parsing**:
    - Absolute and relative indexing ([Gómez-Rodríguez & Vilares, 2018](https://aclanthology.org/D18-1162/)).

## Installation 

To run this code the following requirements are needed:

- [Python >=3.8](https://www.python.org/downloads/), althought the main tests were executed in [Python 3.11](https://www.python.org/downloads/release/python-3110/) (recommended).
- [semantic-dependency-parsing/toolkit](https://github.com/semantic-dependency-parsing/toolkit) installed at [eval/sdp-eval](eval/sdp-eval) to evaluate semantic models and [EVALB](http://pauillac.inria.fr/~seddah/evalb_spmrl2013.tar.gz ) installed and compiled at [eval/EVALB](eval/EVALB) to evaluate constituency models. It is possible to disable the evaluation with the official scripts by setting the `SDP_SCRIPT` and `CON_SCRIPT` to `None` in the [separ/utils/common.py](separ/utils/common.py) script. Then, the evaluation will be performed with Python code (slight variations might occur in some cases).

We provided the [environment.yaml](environment.yaml) to create an [Anaconda](https://anaconda.org/) environment as an alternative to the [requirements.txt](requirements.txt).

```shell 
conda env create -n separ -f environment.yaml
```


> [!WARNING]
> Some functions in the code use the native [concurrent.futures](https://docs.python.org/es/3/library/concurrent.futures.html) package for asynchronous execution. The CPU multithreading acceleration is guaranteed in Linux and MacOS systems, but unexpected behaviors can be experienced in Windows, so we highly suggest using WSL virtualization for Windows users or disabling CPU parallelism setting the variable `NUM_WORKERS` to `1` in the [separ/utils/common.py](separ/utils/common.py) script.




## Data preparation 

To deploy and evaluate our models we relied on well-known treebanks in Dependency, Constituency and Semantic Parsing:

- **Dependency Parsing**: We used various multilingual treebanks in the [CoNLL-U format](https://universaldependencies.org/format.html) publicly available in the [Universal Dependencies](https://universaldependencies.org/) website. 
- **Constituency Parsing**:  PTB ([Marcus et al., 2004](https://aclanthology.org/J93-2004/)) and SPMRL corpus ([Seddah et al., 2011](https://aclanthology.org/volumes/W11-38/)).
- **Semantic Parsing**: SDP ([Oepen et al., 2015](https://aclanthology.org/S15-2153/)) and IWPT ([Bouma et al., 2021](https://aclanthology.org/2021.iwpt-1.15/)) treebanks. 

To pre-process all datasets you need to locate all compressed files ([penn_treebank.zip](treebanks/penn_treebank.zip), [SPMRL_SHARED_2014_NO_ARABIC.zip](treebanks/SPMRL_SHARED_2014_NO_ARABIC.zip), [sdp2014_2015_LDC2016T10.tgz](treebanks/sdp2014_2015_LDC2016T10.tgz) and [iwpt2021stdata.tgz](treebanks/iwpt2021stdata.tgz)) in the [treebanks](treebanks) folder. Then, run (you might need `sudo` privileges and the `zip` library installed):

```shell
cd treebanks && python3 parse-ptb.py && python3 parse-spmrl.py && python3 parse-sdp.py && python3 parse-iwpt.py
```

This will install each treebank in the [treebanks](treebanks/) folder. The final folder structure should look like this:

```
treebanks/
    iwpt-2021/
        ar/
        ...
        uk/
    ptb/
        train.ptb
        dev.ptb
        test.ptb
    sdp-2015/
        dm/
            train.sdp
            dev.sdp
            id.sdp
            ood.sdp
        cs/
        pas/
        psd/
        zh/
    spmrl-2014/
        de/
        ...
        sv/
```

We followed the recommended split for the [PTB](treebanks/ptb) (sections 2-21 for training, 22 for validation and 23 for test). For the [sdp-2015/dm](treebank/sdp-2015/dm), [sdp-2015/pas](treebank/sdp-2015/pas), [sdp-2015/psd](treebank/sdp-2015/psd) and [sdp-2015/cs](treebank/sdp-2015/cs) treebanks we used section 20 for validation (evaluation tests are the in-distribution and out-of-distribution files).


## Usage 

The [docs/examples.ipynb](docs/examples.ipynb) notebook includes some examples of use with the implemented classes and methods to parse and linearize input graphs and train and evaluate customizable neural models. 

## Reproducibility 

To reproduce our experiments we recommend running our models from terminal using the same configuration files provided at [configs](configs/) folder. Training, predicting and evaluating our models can be done from the [run.py](run.py) script. Each parser has a string identifier that is introduced as the first argument of the [run.py](run.py) script. 

| **Name** | **Parser** | **Arguments** |
|:---------|:-----------|:--------------|
| `dep-idx` | Absolute and relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) | `rel` |
| `dep-pos` | PoS-tag relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) | |
| `dep-bracket` | Bracketing encoding ($k$-planar) ([Strzyz et al., 2020](https://aclanthology.org/2020.coling-main.223/)) | `k` | 
| `dep-bit4` | $4$-bit projective  encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)) | |
| `dep-bit7` |  $7$-bit $2$-planar encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)) | | 
| `dep-eager` | Arc-Eager transition-based system ([Nivre & Fernández-González, 2002](https://aclanthology.org/J14-2002/) | `n_stack`, `n_buffer` | 
| `con-idx` | Absolute and relative indexing ([Gómez-Rodríguez & Vilares, 2018](https://aclanthology.org/D18-1162/)) | `rel` | 
| `sdp-idx` | Absolute and relative indexing ([Ezquerro et al., 2024]()) | `rel` | 
| `sdp-bracket` | Bracketing encoding ($k$-planar) ([Ezquerro et al., 2024]()) | `k` | 
| `sdp-bit4k` | $4k$-bit encoding ($k$-planar) ([Ezquerro et al., 2024]()) | `k` | 
| `sdp-bit6k` | $6k$-bit encoding ($k$-planar) ([Ezquerro et al., 2024]()) | `k` | 



### Semantic Dependency Parsing 

Our models were trained on the [SemEval-2015 Task 18](https://alt.qcri.org/semeval2015/task18/) and the [IWPT 2021 Shared Task](https://universaldependencies.org/iwpt21/) datasets and evaluated with the [official SDP toolkit](https://github.com/semantic-dependency-parsing/toolkit) (this must be located at [eval/sdp-eval/](../eval/sdp-eval/) folder). 

The [IWPT 2021 Shared Task](https://universaldependencies.org/iwpt21/) uses the [Enhanced CoNLL format](https://universaldependencies.org/u/overview/enhanced-syntax.html), which is not compatible with the raw [SDP-2015 format](https://alt.qcri.org/semeval2015/task18/index.php?id=data-and-tools) required for evaluation. In order to obtain the same data split to reproduce our experiments and convert the CoNLL files to the SDP format, download both datasets from the official websites ([sdp2014_2015_LDC2016T10.tgz](https://catalog.ldc.upenn.edu/LDC2016T10) and [iwpt2021stdata.tgz](https://universaldependencies.org/iwpt21/data.html)) in the [treebanks/](../treebanks/) folder and then run the [parse-sdp.py](../treebanks/parse-sdp.py) and [parse-iwpt.py](../treebanks/parse-iwpt.py) scripts. At the end, the [treebanks/](../treebanks/) folder must contain the following structure:

```
treebanks/
    iwpt-2021/
        ar/
        ...
        uk/
    sdp-2015/
        cs/
        dm/
        pas/
        psd/
        zh/
    iwpt-2021.tar.gz
    iwpt2021stdata.tgz
    sdp-2015.tar.gz
    sdp2014_2015_LDC2016T10.tgz
```

Each treebank folder (e.g. [dm/](../treebanks/sdp-2015/dm/)) contains the train, dev and test split in the SDP and Enhanced CoNLL format. In case of SDP treebanks, the test split is conformed by the in-distribution ([id.sdp](../treebanks/sdp-2015/dm/id.sdp)) and out-of-distribution files ([ood.sdp](../treebanks/sdp-2015/dm/ood.sdp)).

The [run.py](../run.py) script allows running the training, prediction and evaluation processes with our models. The first argument must specify the algorithm to use and the flag `-c` specifies the configuration file to build the neural encoder (see [configs/](../configs/) folder).

For instance, to train a BiLSTM-based encoder with the $3$-planar bracketing encoding in the DM dataset, run:

```shell
python3 run.py sdp-bracket -c configs/bilstm.ini -p results/dm/ --device=0 -n=3 train \
    --train treebanks/sdp-2015/dm/train.sdp \
    --dev treebanks/sdp-2015/dm/dev.sdp \
    --test treebanks/sdp-2015/dm/id.sdp treebanks/sdp-2015/dm/ood.sdp
```

The encoder can be changed using the other configurations (we provided the encoder with XLM-RoBERTa at [configs/xlm.ini](configs/xlm.ini) and XLNet ([configs/xlnet.ini](configs/xlnet.ini))). 


## Dependency Parsing [on-going]

## Constituency Parsing [on-going]



