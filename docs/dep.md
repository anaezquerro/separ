# Dependency Parsing 

This document provides examples training, evaluating and predicting with dependency parsers. All files passed through input arguments to load data are **CoNLL files**. 

| **Identifier** | **Parser** | **Paper** | **Arguments** |
|:---------|:-----------|:----------|:--------------|
| `dep-idx` | Absolute and relative indexing  | [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/) | `rel` |
| `dep-pos` | PoS-tag relative indexing | [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/) | `gold` |
| `dep-bracket` | Bracketing encoding ($k$-planar) | [Strzyz et al. (2020)](https://aclanthology.org/2020.coling-main.223/) | `k` | 
| `dep-bit4` | $4$-bit projective  encoding | [Gómez-Rodríguez et al. (2023)](https://aclanthology.org/2023.emnlp-main.393/) | `proj` |
| `dep-bit7` |  $7$-bit $2$-planar encoding | [Gómez-Rodríguez et al. (2023)](https://aclanthology.org/2023.emnlp-main.393/) | | 
| `dep-eager` | Arc-Eager system | [Nivre and Fernández-González (2002)](https://aclanthology.org/J14-2002/) | `stack`, `buffer`, `proj` | 
| `dep-biaffine` | Biaffine dependency parser | [Dozat et al. (2016)](https://arxiv.org/abs/1611.01734) | |
| `dep-hexa`   | Hexa-Tagging | [Amini et al. (2023)](https://aclanthology.org/2023.acl-short.124/) | `proj` |


## Training 


- Absolute indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder. Add `--rel` argument as `<specific-args>` to exchange absolute for relative positions.

```shell 
python3 run.py dep-idx -p results/dep-idx-xlnet -c configs/xlnet.ini \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu --num-workers 20
```

- PoS-based relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) with BiLSTMs as encoder. Add `--gold` argument as `<specific-args>` to not predict the PoS-tags but use the gold annotations.
```shell 
python3 run.py dep-pos -p results/dep-pos-bilstm -c configs/bilstm.ini \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu --num-workers 20
```


- Bracketing encoding ([Strzyz et al., 2020](https://aclanthology.org/2020.coling-main.223/)) with $k=2$ with XLM ([Conneau et al., 2019](https://aclanthology.org/2020.acl-main.747/)) as encoder:
```shell 
python3 run.py dep-bracket -k 2 -p results/dep-bracket-xlm -c configs/xlm.ini \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu --num-workers 20
```

- Arc-Eager transition-based system ([Nivre and Fernández-González, 2002](https://aclanthology.org/J14-2002/)) where each state is represented with 1 position of the stack and 2 positions of the buffer. Use XLM ([Conneau et al., 2019](https://aclanthology.org/2020.acl-main.747/)) as encoder:
```shell 
python3 run.py dep-eager --stack 1 --buffer 2 -p results/dep-eager-xlm -c configs/xlm.ini \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu --num-workers 20
```

- Hexa-Tagging ([Amini et al., 2023](https://aclanthology.org/2023.acl-short.124/)) with XLNet as encoder and the *head* pseudo-projective transformation (modes available are `head`, `head+path`, `path`) from [Nivre and Nilsson (2005)](https://aclanthology.org/P05-1013/):
```shell 
python3 run.py dep-hexa -p results/dep-hexa-xlnet -c configs/xlnet.ini --proj head \
    --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu --num-workers 20
```

## Evaluation 
Evaluate now the trained parser at `results/dep-bracket-xlm/parser.pt` with the same test file:
```shell 
python3 run.py dep-bracket -p results/dep-bracket-xlm/parser.pt eval treebanks/english-ewt/test.conllu --batch-size 50
```

## Prediction 
Predict with the trained parser at `results/dep-hexa-xlnet`:
```shell 
python3 run.py dep-hexa -p results/dep-hexa-xlnet/parser.pt predict \
     treebanks/english-ewt/test.conllu results/dep-hexa-xlnet/pred.conllu --batch-size 50
```