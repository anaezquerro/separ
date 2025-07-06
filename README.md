# :seedling: [SePar](https://github.com/anaezquerro/separ): [Se]()quence Labeling algorithms for [Par]()sing

Hi :wave: This is a [Python](https://www.python.org/) implementation for sequence-labeling algorithms for Dependency, Constituency and Graph Parsing.


- **Dependency Parsing**:
    - Absolute and relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)).
    - PoS-tag relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)).
    - Bracketing encoding ($k$-planar) ([Strzyz et al., 2020](https://aclanthology.org/2020.coling-main.223/)).
    - $4$-bit projective  encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)).
    - $7$-bit $2$-planar encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393/)).
    - Hierarchical bracketing encoding ([Ezquerro et al., 2025](https://arxiv.org/abs/2505.11693), **OFFICIAL**).
    - Hexa-Tagging projective encoding ([Amini et al., 2023](https://aclanthology.org/2023.acl-short.124/)).
    - Arc-Eager transition-based system ([Nivre and Fernández-González, 2002](https://aclanthology.org/J14-2002/)).
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


It is also the official repository of the following papers:

- *Dependency Graph Parsing as Sequence Labeling* ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)).
- *Hierarchical Bracketing Encodings for Dependency Parsing as Tagging* ([Ezquerro et al., 2025](https://arxiv.org/abs/2505.11693)).


Please, feel free to [reach out](mailto:ana.ezquerro@udc.es) if you want to collaborate or include  additional parsers to [SePar](https://github.com/anaezquerro/separ)!



## Installation 

This code was tested in [Python >=3.8](https://www.python.org/downloads/), although we recommend using [Python >=3.12](https://www.python.org/downloads/release/python-3110/) in a GPU system with NVIDIA drivers (>=535) and CUDA (>=12.4) installed. Use the [requirements.txt](requirements.txt) to download the dependencies in an existing environment or the [environment.yaml](environment.yaml) to create an Anaconda environment.

```shell
pip install -r requirements.txt
```
or
```shell 
conda env create -n separ -f environment.yaml
```

## Usage 

You can train, evaluate and predict different parsers from terminal with the [run.py](run.py) script. Each parser has a string identifier that is introduced as the first argument of [run.py](run.py). The following table shows the available parsers and their configuration (modifiable through terminal arguments).


| **Identifier** | **Parser** | **Paper** | **Arguments** | **Default** | 
|:---------|:-----------|:----------|:--------------|:------------------|
| `dep-idx` | Absolute and relative indexing  | [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/) | `rel`$\in$`{true, false}` | `false` | 
| `dep-pos` | PoS-tag relative indexing | [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/) |  | | 
| `dep-bracket` | Bracketing encoding ($k$-planar) | [Strzyz et al. (2020)](https://aclanthology.org/2020.coling-main.223/) | `k`$\in\mathbb{N}$ | `1` | 
| `dep-bit4` | $4$-bit projective  encoding | [Gómez-Rodríguez et al. (2023)](https://aclanthology.org/2023.emnlp-main.393/) | `proj`$\in$`{None, head, head+path, path}` | `None` | 
| `dep-bit7` |  $7$-bit $2$-planar encoding | [Gómez-Rodríguez et al. (2023)](https://aclanthology.org/2023.emnlp-main.393/) | | 
| `dep-hexa`   | Hexa-Tagging | [Amini et al. (2023)](https://aclanthology.org/2023.acl-short.124/) | `proj`$\in$`{head, head+path, path}` | `head` | 
| `dep-hier` | Hierarchical Bracketing | [Ezquerro et al., (2025)](https://arxiv.org/abs/2505.11693) | `variant`$\in$ `{proj, head, head+path, path, nonp}`| `proj` | 
| `dep-eager` | Arc-Eager system | [Nivre and Fernández-González (2002)](https://aclanthology.org/J14-2002/) | `stack`$\in\mathbb{N}$, `buffer`$\in\mathbb{N}$, `proj`$\in$`{None, head, head+path, path}` | `1`, `1`, `None`| 
| `dep-biaffine` | Biaffine dependency parser | [Dozat & Manning (2017)](https://arxiv.org/abs/1611.01734) | |
| `con-idx` | Absolute and relative indexing  | [Gómez-Rodríguez and Vilares (2018)](https://aclanthology.org/D18-1162/) | `rel`$\in$`{true, false}` |  `false` | 
| `con-tetra` | Tetra-Tagging | [Kitaev and Klein (2020)](https://aclanthology.org/2020.acl-main.557/) | | | 
| `sdp-idx` | Absolute and relative indexing  | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `rel`$\in$`{true, false}` |  `false` |
| `sdp-bracket` | Bracketing encoding ($k$-planar) | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `2` | 
| `sdp-bit4k` | $4k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `3` |
| `sdp-bit6k` | $6k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `3` |
| `sdp-cov`| Covington | [Covington (2001)](https://ai1.ai.uga.edu/mc/dparser/dgpacmnew.pdf) | | | 
| `sdp-biaffine` | Biaffine graph parser | [Dozat & Manning (2018)](https://aclanthology.org/P18-2077/) | || 


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

**W&B logging**: [SePar](https://github.com/anaezquerro/separ) also allows model debugging with [wandb](https://wandb.ai/site/). Please. follow [these instructions](https://docs.wandb.ai/quickstart/) to create and account and connect it with your local installation. Note that [SePar](https://github.com/anaezquerro/separ) still works without a [wandb](https://wandb.ai/site/) account.

### Distributed training

[SePar](https://github.com/anaezquerro/separ) supports distributed training with [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) by running the script [run.py](run.py) with `torchrun`. Use the `CUDA_VISIBLE_DEVICES` variable to hide specific GPUs.

```shell
CUDA_VISIBLE_DEVICES=<devices> torchrun --nproc_per_node <num-devices> \
    run.py <parser-identifier> <specific-args> \
    -c <conf> (--load <pt-path> --seed <seed>) \
    train --train <train-path> --dev <dev-path> --test <test-paths> \
    -o <output-folder> (--run-name <run-name>)
```

where `<devices>` is the list of GPU identifiers (separated by comma) and `<num-devices>` is the number of GPUs used.


> [!WARNING]  
> As introduced in [this tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), FSDP2 requires manually specifying which modules or layers are sharded between GPUs for a better parameter distribution. In [separ/utils/shard.py](separ/utils/shard.py) we include a function `recursive_shard()` which only shards large Transformer layers (specifically, those corresponding to pretrained models included in the [configs](configs/) folder). We suggest manually adding more layers when training with other LLMs. Do not hesitate to [reach us](mailto:ana.ezquerro@udc.es) if you need any help!

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

