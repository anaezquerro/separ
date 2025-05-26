# Semantic Parsing 


This document provides examples training, evaluating and predicting with constituency parsers. All files passed through input arguments to load data are **EnhancedCoNLL** or **SDP files** (see [docs/sample.sdp](docs/sample.sdp) for an example).


| **Identifier** | **Parser** | **Paper** | **Arguments** | **Default** | 
|:---------|:-----------|:----------|:--------------|:------------------|
| `sdp-idx` | Absolute and relative indexing  | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `rel`$\in$`{true, false}` |  `false` |
| `sdp-bracket` | Bracketing encoding ($k$-planar) | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `2` | 
| `sdp-bit4k` | $4k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `3` |
| `sdp-bit6k` | $6k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k`$\in\mathbb{N}$ | `3` |
| `sdp-cov`| Covington | [Covington (2001)](https://ai1.ai.uga.edu/mc/dparser/dgpacmnew.pdf) | | | 
| `sdp-biaffine` | Biaffine graph parser | [Dozat & Manning (2018)](https://aclanthology.org/P18-2077/) | || 

### Training

- Relative indexing ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder. Remove the `--rel` argument as to exchange relative for absolute positions. See that two test files can be added. 

```shell 
python3 run.py sdp-idx --rel -c configs/xlnet.ini \
    train --train treebanks/sdp-2015/dm/train.sdp \
    --dev treebanks/sdp-2015/dm/dev.sdp \
    --test treebanks/sdp-2015/dm/id.sdp treebanks/sdp-2015/dm/ood.sdp \
    --output-folder results/sdp-idx-xlnet
```

- $4k$-bit encoding ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder and $k=4$.
```shell 
python3 run.py sdp-bit4k -k=4 -c configs/xlnet.ini \
    train --train treebanks/iwpt-2021/en/train.conllu \
    --dev treebanks/iwpt-2021/en/dev.conllu \
    --test treebanks/iwpt-2021/en/test.conllu \
    --output-folder results/sdp-bit4k-xlnet
```

## Evaluation 
Evaluate now the trained parser at `results/sdp-idx-xlnet/parser.pt` with the same test file:
```shell 
python3 run.py sdp-idx --load results/sdp-idx-xlnet/parser.pt \
    eval treebanks/sdp-2015/dm/id.sdp --batch-size 150
```

## Prediction 
Predict with the trained parser at  `results/sdp-idx-xlnet/parser.pt`:
```shell 
python3 run.py sdp-idx --load results/sdp-idx-xlnet/parser.pt predict \
    treebanks/sdp-2015/dm/id.sdp results/sdp-idx-xlnet/pred.sdp --batch-size 150
```
