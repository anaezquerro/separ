# Constituency Parsing

This document provides examples training, evaluating and predicting with constituency parsers. All files passed through input arguments to load data are **PTB bracket files** (see [docs/sample.ptb](docs/sample.ptb) for an example).

| **Identifier** | **Parser** | **Paper** | **Arguments** | **Default** | 
|:---------|:-----------|:----------|:--------------|:------------------|
| `con-idx` | Absolute and relative indexing  | [Gómez-Rodríguez and Vilares (2018)](https://aclanthology.org/D18-1162/) | `rel`$\in$`{true, false}` |  `false` | 
| `con-tetra` | Tetra-Tagging | [Kitaev and Klein (2020)](https://aclanthology.org/2020.acl-main.557/) | | | 



### Training
- Relative indexing ([Gómez-Rodríguez and Vilares, 2018](https://aclanthology.org/D18-1162/) ) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder.

```shell 
python3 run.py con-idx --rel -c configs/xlnet.ini \
    train --train $DATA_FOLDER/ptb/train.ptb \
    --dev $DATA_FOLDER/ptb/dev.ptb \
    --test $DATA_FOLDER/ptb/test.ptb \
    --output-folder $RESULTS_FOLDER/con-rel-xlnet
```

- Tetra-tagging ([Kitaev and Klein, 2020](https://aclanthology.org/2020.acl-main.557/)) with XLNet  ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder.
```shell 
python3 run.py con-tetra -c configs/xlnet.ini \
    train --train $DATA_FOLDER/ptb/train.ptb \
    --dev $DATA_FOLDER/ptb/dev.ptb \
    --test $DATA_FOLDER/ptb/test.ptb \
    --output-folder $RESULTS_FOLDER/con-tetra-xlnet 
```

## Evaluation 
Evaluate now the trained parser at `$RESULTS_FOLDER/con-tetra-xlnet/parser.pt` with the same test file:
```shell 
python3 run.py con-tetra --load $RESULTS_FOLDER/con-tetra-xlnet/parser.pt \
    eval $DATA_FOLDER/ptb/test.ptb --batch-size 50
```

## Prediction 
Predict with the trained parser at  `$RESULTS_FOLDER/con-tetra-xlnet/parser.pt`:
```shell 
python3 run.py con-tetra --load $RESULTS_FOLDER/con-tetra-xlnet/parser.pt predict \
    $DATA_FOLDER/ptb/test.ptb $RESULTS_FOLDER/con-tetra-xlnet/pred.ptb --batch-size 50
```

## Reproducibility 

### *Bringing Emerging Architectures to Sequence Labeling in NLP* ([Ezquerro et al., 2025c](https://arxiv.org/abs/2509.25918))

[Ezquerro et al. (2025c)](https://arxiv.org/abs/2509.25918) conducted experiments on tree indexing ([Gómez-Rodríguez and Vilares, 2018](https://aclanthology.org/D18-1162/)) and tetra-tagging ([Kitaev and Klein, 2020](https://aclanthology.org/2020.acl-main.557/)) using a diffusion and adversarial tagger. 

> [!NOTE]  
> For the diffusion model, note that the parameters of `[dec_conf]` are loaded from the configuration file, while for the adversarial tagger, the `[dis_conf]` is used. To call the diffusion (`dit`) or adversarial (`gat`) model, use the `dit` and `gat` variants of each graph encoding (introducing the flag between the `grp` prefix and encoding suffix). 

An example to run tetratagging on the PTB with the adversarial tagger:
```shell 
python3 run.py con-gat-tetra -c configs/xlnet.ini \
    train --train $DATA_FOLDER/ptb/train.ptb \
          --dev $DATA_FOLDER/ptb/dev.ptb \
          --test $DATA_FOLDER/ptb/test.ptb 
          --output-folder $RESULTS_FOLDER/con-gat-tetra
```

