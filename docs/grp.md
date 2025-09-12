# Graph Parsing 


This document provides examples training, evaluating and predicting with constituency parsers. All files passed through input arguments to load data are **EnhancedCoNLL** or **SDP files** (see [docs/sample.sdp](docs/sample.sdp) for an example).


| **Identifier** | **Parser** | **Paper** | **Arguments** | **Default** | 
|:---------|:-----------|:----------|:--------------|:------------------|
| `grp-idx` | Absolute and relative indexing  | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `rel`$\in$`{true, false}` |  `false` |
| `grp-bracket` | Bracketing encoding ($k$-planar) | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `2` | 
| `grp-hier` | Hierarchical bracketing encoding | [Ezquerro et al., (2025)](https://arxiv.org/abs/2509.09388) |
| `grp-bit4k` | $4k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `3` |
| `grp-bit6k` | $6k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `3` |
| `grp-hier` | Hierarchical bracketing encoding | [Ezquerro et al., (2025)](https://arxiv.org/abs/2509.09388) | |
| `grp-cov`| Covington | [Covington (2001)](https://ai1.ai.uga.edu/mc/dparser/dgpacmnew.pdf) | | | 
| `grp-biaffine` | Biaffine graph parser | [Dozat & Manning (2018)](https://aclanthology.org/P18-2077/) | || 

### Training

- Relative indexing ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder. Remove the `--rel` argument as to exchange relative for absolute positions. See that two test files can be added. 

```shell 
python3 run.py grp-idx --rel -c configs/xlnet.ini \
    train --train $DATA_FOLDER/grp-2015/dm/train.sdp \
    --dev $DATA_FOLDER/grp-2015/dm/dev.sdp \
    --test $DATA_FOLDER/grp-2015/dm/id.sdp $DATA_FOLDER/grp-2015/dm/ood.sdp \
    --output-folder $RESULTS_FOLDER/grp-idx-xlnet
```

- $4k$-bit encoding ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder and $k=4$.
```shell 
python3 run.py grp-bit4k -k=4 -c configs/xlnet.ini \
    train --train $DATA_FOLDER/iwpt-2021/en/train.conllu \
    --dev $DATA_FOLDER/iwpt-2021/en/dev.conllu \
    --test $DATA_FOLDER/iwpt-2021/en/test.conllu \
    --output-folder $RESULTS_FOLDER/grp-bit4k-xlnet
```

## Evaluation 
Evaluate now the trained parser at `$RESULTS_FOLDER/grp-idx-xlnet/parser.pt` with the same test file:
```shell 
python3 run.py grp-idx --load $RESULTS_FOLDER/grp-idx-xlnet/parser.pt \
    eval $DATA_FOLDER/grp-2015/dm/id.sdp --batch-size 150
```

## Prediction 
Predict with the trained parser at  `$RESULTS_FOLDER/grp-idx-xlnet/parser.pt`:
```shell 
python3 run.py grp-idx --load $RESULTS_FOLDER/grp-idx-xlnet/parser.pt predict \
    $DATA_FOLDER/grp-2015/dm/id.sdp $RESULTS_FOLDER/grp-idx-xlnet/pred.sdp --batch-size 150
```
