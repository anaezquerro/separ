# Constituency Parsing

This document provides examples training, evaluating and predicting with constituency parsers. All files passed through input arguments to load data are **PTB bracket files** (see [docs/sample.ptb](docs/sample.ptb) for an example).


| **Identifier** | **Parser** | **Paper** | **Arguments** |
|:---------|:-----------|:----------|:--------------|
| `con-idx` | Absolute and relative indexing  | [Gómez-Rodríguez and Vilares (2018)](https://aclanthology.org/D18-1162/) | `rel` | 
| `con-tetra` | Tetra-Tagging | [Kitaev and Klein (2020)](https://aclanthology.org/2020.acl-main.557/) | |



### Training
- Absolute indexing ([Gómez-Rodríguez and Vilares, 2018](https://aclanthology.org/D18-1162/) ) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder. Add `--rel` argument as `<specific-args>` to exchange absolute for relative positions.

```shell 
python3 run.py con-idx -p results/con-idx-xlnet -c configs/xlnet.ini \
    train --train treebanks/ptb/train.ptb \
    --dev treebanks/ptb/dev.ptb \
    --test treebanks/ptb/test.ptb --num-workers 20
```

- Tetra-tagging ([Kitaev and Klein, 2020](https://aclanthology.org/2020.acl-main.557/)) with XLNet  ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder.
```shell 
python3 run.py con-tetra -p results/con-tetra-xlnet -c configs/xlnet.ini \
    train --train treebanks/ptb/train.ptb \
    --dev treebanks/ptb/dev.ptb \
    --test treebanks/ptb/test.ptb --num-workers 20
```

## Evaluation 
Evaluate now the trained parser at `results/con-tetra-xlnet/parser.pt` with the same test file:
```shell 
python3 run.py con-tetra -p results/con-tetra-xlnet/parser.pt eval treebanks/ptb/test.ptb --batch-size 50
```

## Prediction 
Predict with the trained parser at  `results/con-tetra-xlnet/parser.pt`:
```shell 
python3 run.py con-tetra -p results/con-tetra-xlnet/parser.pt predict \
    treebanks/ptb/test.ptb results/con-tetra-xlnet/pred.ptb --batch-size 50
```
