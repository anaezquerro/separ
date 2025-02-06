# Semantic Parsing 


This document provides examples training, evaluating and predicting with constituency parsers. All files passed through input arguments to load data are **EnhancedCoNLL** or **SDP files** (see [docs/sample.sdp](docs/sample.sdp) for an example).

| **Identifier** | **Parser** | **Paper** | **Arguments** |
|:---------|:-----------|:----------|:--------------|
| `sdp-idx` | Absolute and relative indexing  | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `rel` | 
| `sdp-bracket` | Bracketing encoding ($k$-planar) | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k` | 
| `sdp-bit4k` | $4k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k` | 
| `sdp-bit6k` | $6k$-bit encoding | [Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) | `k` | 


### Training

- Relative indexing ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder. Remove the `--rel` argument as to exchange relative for absolute positions. See that two test files can be added. 

```shell 
python3 run.py sdp-idx --rel -p results/sdp-idx-xlnet -c configs/xlnet.ini \
    train --train treebanks/sdp-2015/dm/train.sdp \
    --dev treebanks/sdp-2015/dm/dev.sdp \
    --test treebanks/sdp-2015/dm/id.sdp treebanks/sdp-2015/dm/ood.sdp \
    --num-workers 20
```

- $4k$-bit encoding ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder and $k=4$.
```shell 
python3 run.py sdp-bit4k -k=4 -p results/sdp-bit4k-xlnet -c configs/xlnet.ini \
    train --train treebanks/iwpt-2021/en/train.conllu \
    --dev treebanks/iwpt-2021/en/dev.conllu \
    --test treebanks/iwpt-2021/en/test.conllu \
    --num-workers 20
```

## Evaluation 
Evaluate now the trained parser at `results/sdp-idx-xlnet/parser.pt` with the same test file:
```shell 
python3 run.py sdp-idx -p results/sdp-idx-xlnet/parser.pt eval treebanks/sdp-2015/dm/id.sdp --batch-size 150
```

## Prediction 
Predict with the trained parser at  `results/sdp-idx-xlnet/parser.pt`:
```shell 
python3 run.py sdp-idx -p results/sdp-idx-xlnet/parser.pt predict \
    treebanks/sdp-2015/dm/id.sdp results/sdp-idx-xlnet/pred.sdp --batch-size 150
```
