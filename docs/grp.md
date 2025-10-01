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

## Reproducibility
### *Dependency Graph Parsing as Sequence Labeling* ([Ezquerro et al., 2024](https://aclanthology.org/2024.emnlp-main.659/))

[Ezquerro et al. (2024)](https://aclanthology.org/2024.emnlp-main.659/) conducted experiments on the SDP ([Oepen et al., 2015](https://aclanthology.org/S15-2153/)) and IWPT ([Bouma et al., 2021](https://aclanthology.org/2021.iwpt-1.15/)) datasets and used the XLNet for English treebanks and XLM for non-English treebanks as encoder. The configuration used to run their experiments is included in the files [configs/xlnet.ini](../configs/xlnet.ini) and [configs/xlm.ini](../configs/xlm.ini).

- **SDP**  ([Oepen et al., 2015](https://aclanthology.org/S15-2153/)): For the English (DM, PAS, PSD) and Czech (PAS) annotations, the suggested training and validation splits were used (sentences with ID `#220*`), and the in-distribution and out-of-distribution sets were used for model evaluation. On the Chinese (PSD) annotations, the sentences with ID `#202*` were used for validation, and only the in-distribution set was used for evaluation.

- **IWPT** ([Bouma et al., 2021](https://aclanthology.org/2021.iwpt-1.15/)): All the sets of each language were merged to obtain a single train-dev-test split for each language. 

The neural taggers used do not activate the `--join-rels` flag, meaning that the arc label was predicted with an independent module (not in tagging mode). For instance, the English DM treebank with the bracketing encoding $k=2$ was trained and evaluated as:

```shell
python3 run.py grp-bracket -k 2 -c configs/xlnet.ini \
    train --train $DATA_FOLDER/sdp-2015/dm/train.sdp \
          --dev $DATA_FOLDER/sdp-2015/dm/dev.sdp \
          --test $DATA_FOLDER/sdp-2015/dm/id.sdp $DATA_FOLDER/sdp-2015/dm/ood.sdp \
          --output-folder $RESULTS_FOLDER/test
```

### *Hierarchical Bracketing Encodings Work for Dependency Graphs* ([Ezquerro et al., 2025b](https://www.arxiv.org/abs/2509.09388))

[Ezquerro et al. (2025b)](https://www.arxiv.org/abs/2509.09388) expanded the hierarchical bracketing encoding from dependency trees ([Ezquerro et al., 2025a](https://aclanthology.org/2025.acl-long.903/)) to dependency graphs. It was evaluated on the SDP ([Oepen et al., 2015](https://aclanthology.org/S15-2153/)) and some IWPT  ([Bouma et al., 2021](https://aclanthology.org/2021.iwpt-1.15/)) treebanks (check the original paper for the specifications).

To reproduce the results, we provide here an execution example to train the hierarchical bracketing encoding with the English DM treebank:

```shell
python3 run.py grp-hier -c configs/xlnet.ini \
    train --train $DATA_FOLDER/sdp-2015/dm/train.sdp \
          --dev $DATA_FOLDER/sdp-2015/dm/dev.sdp \
          --test $DATA_FOLDER/sdp-2015/dm/id.sdp $DATA_FOLDER/sdp-2015/dm/ood.sdp \
          --output-folder $RESULTS_FOLDER/test
```


### *Bringing Emerging Architectures to Sequence Labeling in NLP* ([Ezquerro et al., 2025c](https://arxiv.org/abs/2509.25918))

[Ezquerro et al. (2025c)](https://arxiv.org/abs/2509.25918) also evaluated graph encodings using diffusion and adversarial taggers. 

> [!NOTE]  
> For the diffusion model, note that the parameters of `[dec_conf]` are loaded from the configuration file, while for the adversarial tagger, the `[dis_conf]` is used. To call the diffusion (`dit`) or adversarial (`gat`) model, use the `dit` and `gat` variants of each graph encoding (introducing the flag between the `grp` prefix and encoding suffix). 

To reproduce their results, the flag `--join-rels` must be explictly introduced, since their experiments *also* predict the arc labels as a tagging task. For instance, to train the bracketing encoding with $k=2$ using the diffusion model:

```shell
python3 run.py grp-dit-bracket -k 2 --join-rels configs/xlnet.ini \
    train --train $DATA_FOLDER/sdp-2015/dm/train.sdp \
          --dev $DATA_FOLDER/sdp-2015/dm/dev.sdp \
          --test $DATA_FOLDER/sdp-2015/dm/id.sdp $DATA_FOLDER/sdp-2015/dm/ood.sdp \
          --output-folder $RESULTS_FOLDER/test
```
