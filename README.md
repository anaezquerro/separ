# :seedling: Parsing as Sequence Labeling

Hi :wave: This is a [Python](https://www.python.org/) implementation for sequence labeling algorithms (and others!) for Dependency, Constituency and Semantic Parsing.


- **Dependency Parsing**:
    - Absolute and relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)).
    - PoS-tag relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)).
    - Bracketing encoding ($k$-planar) ([Strzyz et al., 2020](https://aclanthology.org/2020.coling-main.223/)).
    - Arc-Eager transition-based system ([Nivre and Fernández-González, 2002](https://aclanthology.org/J14-2002/)).
    - $4$-bit projective  encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)).
    - $7$-bit $2$-planar encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)).
    - Hexa-Tagging projective encoding ([Amini et al., 2023](https://aclanthology.org/2023.acl-short.124/)).
    - Biaffine dependency parser ([Dozat & Manning, 2017](https://arxiv.org/abs/1611.01734)).
- **Semantic Dependency Parsing**:
    - Absolute and relative indexing ([Ezquerro et al., 2024](https://arxiv.org/abs/2410.17972), **OFFICIAL**).
    - Bracketing encoding ($k$-planar) ([Ezquerro et al., 2024](https://arxiv.org/abs/2410.17972), **OFFICIAL**).
    - $4k$-bit encoding ([Ezquerro et al., 2024](https://arxiv.org/abs/2410.17972), **OFFICIAL**).
    - $6k$-bit encoding ([Ezquerro et al., 2024](https://arxiv.org/abs/2410.17972), **OFFICIAL**).
    - Covington graph parser ([Covington, 2001](https://ai1.ai.uga.edu/mc/dparser/dgpacmnew.pdf)).
    - Biaffine semantic parser ([Dozat & Manning, 2018](https://aclanthology.org/P18-2077/)).
- **Constituency Parsing**:
    - Absolute and relative indexing ([Gómez-Rodríguez and Vilares, 2018](https://aclanthology.org/D18-1162/)).
    - Tetra-Tagging ([Kitaev and Klein, 2020](https://aclanthology.org/2020.acl-main.557/)).

Please, feel free to [reach out](mailto:ana.ezquerro@udc.es) if you want to collaborate or include  additional parsers to [separ](https://github.com/anaezquerro/separ)!

## Installation 

This code was tested in [Python >=3.8](https://www.python.org/downloads/), although we recommend using [Python >=3.11](https://www.python.org/downloads/release/python-3110/) in a GPU system with NVIDIA drivers (>=535) and CUDA (>=12.4) installed. Use the [requirements.txt](requirements.txt) to download the dependencies in an existing environment or the [environment.yaml](environment.yaml) to create an Anaconda environment.

```shell
pip install -r requirements.txt
```
or
```shell 
conda env create -n separ -f environment.yaml
```

## Data preparation 

To deploy and evaluate our models we relied on well-known treebanks in Dependency, Constituency and Semantic Parsing:

- **Dependency Parsing**: We used various multilingual treebanks in the [CoNLL-U format](https://universaldependencies.org/format.html) publicly available in the [Universal Dependencies](https://universaldependencies.org/) website. 
- **Constituency Parsing**:  PTB ([Marcus et al., 2004](https://aclanthology.org/J93-2004/)) and SPMRL corpus ([Seddah et al., 2011](https://aclanthology.org/volumes/W11-38/)).
- **Semantic Parsing**: SDP ([Oepen et al., 2015](https://aclanthology.org/S15-2153/)) and IWPT ([Bouma et al., 2021](https://aclanthology.org/2021.iwpt-1.15/)) treebanks. 

To pre-process all datasets you need to locate all compressed files ([penn_treebank.zip](treebanks/penn_treebank.zip), [SPMRL_SHARED_2014_NO_ARABIC.zip](treebanks/SPMRL_SHARED_2014_NO_ARABIC.zip), [sdp2014_2015_LDC2016T10.tgz](treebanks/sdp2014_2015_LDC2016T10.tgz) and [iwpt2021stdata.tgz](treebanks/iwpt2021stdata.tgz)) in the [treebanks](treebanks) folder. Then, run from the [treebanks](treebanks) folder  (you might need `sudo` privileges and the `zip` library installed) these scripts:

- [parse-ptb.py](treebanks/parse-ptb.py): To split the PTB into three sets (train, development and test) in the raw bracketing format. We follow the recommended split: sections 2-21 for training, 22 for validation and 23 for test. The result should be something like this:
```
treebanks/
    ptb/
        train.ptb
        dev.ptb.
        test.ptb 
```
- [parse-spmrl.py](treebanks/parse-spmrl.py): To create a subfolder per language of the SPMRL multilingual dataset. The result should be something like this:
```
treebanks/
    spmrl-2014/
        de/ 
            train.ptb
            dev.ptb
            test.ptb
        ...
        sv/
```
- [parse-sdp.py](treebanks/parse-sdp.py): To create a subfolder per treebank of the SDP dataset. For the DM (English), PAS (English), PSD (English) and PSD (Czech) treebanks we used section 20 for validation (evaluation tests are the in-distribution and out-of-distribution files).
```
treebanks/
    sdp-2015/
        dm/
            train.sdp
            dev.sdp
            id.sdp
            ood.sdp
        ...
        zh/
```
- [parse-iwpt.py](treebanks/parse-iwpt.py): To create subfolder per language in the IWPT dataset. 
```
treebanks/
    iwpt-2021/
        ar-padt/
            train.conllu
            dev.conllu
            test.conllu
        ...
        uk-iu/
```

## Usage 

You can train, evaluate and predict different parsers from terminal with [run.py](run.py). Each parser has a string identifier that is introduced as the first argument of the [run.py](run.py) script. The following table shows the parsers available with their corresponding paper and the 
proper arguments that can be introduced:


| **Identifier** | **Parser** | **Paper** | **Arguments** |
|:---------|:-----------|:----------|:--------------|
| `dep-idx` | Absolute and relative indexing  | [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/) | `rel`$\in$`{true, false}` |
| `dep-pos` | PoS-tag relative indexing | [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/) |  |
| `dep-bracket` | Bracketing encoding ($k$-planar) | [Strzyz et al. (2020)](https://aclanthology.org/2020.coling-main.223/) | `k`$\in\mathbb{N}$ | 
| `dep-bit4` | $4$-bit projective  encoding | [Gómez-Rodríguez et al. (2023)](https://aclanthology.org/2023.emnlp-main.393/) | `proj`$\in$`{None, head, head+path, path}` |
| `dep-bit7` |  $7$-bit $2$-planar encoding | [Gómez-Rodríguez et al. (2023)](https://aclanthology.org/2023.emnlp-main.393/) | | 
| `dep-eager` | Arc-Eager system | [Nivre and Fernández-González (2002)](https://aclanthology.org/J14-2002/) | `stack`$\in\mathbb{N}$, `buffer`$\in\mathbb{N}$, `proj`$\in$`{None, head, head+path, path}` | 
| `dep-biaffine` | Biaffine dependency parser | [Dozat & Manning (2017)](https://arxiv.org/abs/1611.01734) | |
| `dep-hexa`   | Hexa-Tagging | [Amini et al. (2023)](https://aclanthology.org/2023.acl-short.124/) | `proj`$\in$`{None, head, head+path, path}` |
| `con-idx` | Absolute and relative indexing  | [Gómez-Rodríguez and Vilares (2018)](https://aclanthology.org/D18-1162/) | `rel`$\in$`{true, false}` | 
| `con-tetra` | Tetra-Tagging | [Kitaev and Klein (2020)](https://aclanthology.org/2020.acl-main.557/) | |
| `sdp-idx` | Absolute and relative indexing  | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `rel`$\in$`{true, false}` | 
| `sdp-bracket` | Bracketing encoding ($k$-planar) | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | 
| `sdp-bit4k` | $4k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | 
| `sdp-bit6k` | $6k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | 
| `sdp-cov`| Covington | [Covington (2001)](https://ai1.ai.uga.edu/mc/dparser/dgpacmnew.pdf) | |
| `sdp-biaffine` | Biaffine graph parser | [Dozat & Manning (2018)](https://aclanthology.org/P18-2077/) | |


### Training
To train a parser from scratch, the [run.py](run.py) script should follow this syntax:

```shell
python3 run.py <parser-identifier> <specific-args> \
    -c <conf> -d <device> (--load <pt-path> --seed <seed>) \
    train --train <train-path> --dev <dev-path> --test <test-paths> \
    -o <output-folder> (--run-name <run-name>)
```

where:
- `<parser-identifier>` is the identifier specified in the table above (e.g. `dep-idx`),
- `<specific-args>` are the specific arguments of each parser (e.g. `--rel` for `dep-idx`),
- `<conf>` is the model configuration file (see some examples in [configs](configs/) folder),
- `<device>` is the CUDA integer device,
- `<train-path>`, `<dev-path>` and `<test-paths>` are the paths to the training, development and test sets (multiple test paths are possible).
- `<output-folder>` is a folder to store the training results (including the `parser.pt` file).

And optionally:
- `<pt-path>`: Whether to load the parser from an existing `.pt` file. 
- `<seed>`: Specify other seed value. By default, this code always uses the seed  `123`.
- `<run-name>`: [wandb](https://wandb.ai/site/) identifier.

**Distributed training**: This library supports distributed training by setting the argument `-d=-1` and running the script [run.py](run.py) with `torchrun`. Use the `CUDA_VISIBLE_DEVICES` variable to hide specific GPUs (by default, `-d=-1` will use all GPUs available). 

**W&B logging**: This library also allows model debugging with [wandb](https://wandb.ai/site/). Please. follow [these instructions](https://docs.wandb.ai/quickstart/) to create and account and connect it with your local installation. Note that [separ](https://github.com/anaezquerro/separ) still works without a [wandb](https://wandb.ai/site/) account.



### Evaluation
Evaluation with a trained parser is also performed with the [run.py](run.py) script.

```shell 
python3 run.py <parser-identifier> <specific-args> --load <pt-path> -c <conf> -d <device> 
    eval <input> (--output <output> --batch-size <batch-size>)
```

where:
- `<parser-identifier>` is the identifier specified in the table above (e.g. `dep-idx`),
- `<specific-args>` are the specific arguments of each parser (e.g. `--rel` for `dep-idx`),
- `<pt-path>` is the path where the parser has been stored (e.g. the `parser.pt` file created after training).
- `<conf>` is the model configuration file (see some examples in [configs](configs/) folder),
- `<device>` is the CUDA integer device (e.g. `0`),
- `<input>` is the annotated file to perform the evaluation.

And optionally:
- `<output>`: Folder to store the result metric.
- `<batch-size>`: Inference batch size. By default is set to 100.

### Prediction
Prediction with a trained parser is also conducted from the [run.py](run.py) script. 
```shell 
python3 run.py <parser-identifier> <specific-args> --load <pt-path> -c <conf> -d <device> \
    predict <input> <output> (--batch-size <batch-size>)
```
where:
- `<parser-identifier>` is the identifier specified in the table above (e.g. `dep-idx`),
- `<specific-args>` are the specific arguments of each parser (e.g. `--rel` for `dep-idx`),
- `<pt-path>` is the path where the parser has been stored (e.g. the `parser.pt` file created after training).
- `<conf>` is the model configuration file (see some examples in [configs](configs/) folder),
- `<device>` is the CUDA integer device (e.g. `0`),
- `<input>` is the annotated file to perform the evaluation.
- `<output>` is the file to store the predicted file.

And optionally:
- `<batch-size>`: Inference batch size. By default is set to 100.


## Examples

Check the [docs](docs/) folder for specific examples running different dependency ([docs/dep.md](docs/dep.md)), constituency ([docs/con.md](docs/con.md)) and semantic ([docs/sdp.md](docs/sdp.md)) parsers. The [docs/examples.ipynb](docs/examples.ipynb) notebook includes some examples of how to use the implemented classes and methods to parse and linearize input graphs/trees.

