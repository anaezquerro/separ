# Dependency Parsing 

This document provides examples training, evaluating and predicting with dependency parsers. All files passed through input arguments to load data are **CoNLL files**. 

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

## Training 


- Absolute indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder. Add `--rel` argument as `<specific-args>` to exchange absolute for relative positions.

```shell 
python3 run.py dep-idx -c configs/xlnet.ini -d 0\
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu \
    --output-folder results/dep-abs-xlnet
```

Or for multi-GPU training:
```shell 
CUDA_VISIBLE_DEVICES="0,2,3" torchrun --nproc_per_node=3 run.py dep-idx -c configs/xlnet.ini \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu \
    --output-folder results/dep-abs-xlnet
```


- PoS-based relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) with BiLSTMs as encoder.
```shell 
python3 run.py dep-pos -c configs/bilstm.ini -d 0 \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu \
    --output-folder results/dep-pos-bilstm 
```


- Bracketing encoding ([Strzyz et al., 2020](https://aclanthology.org/2020.coling-main.223/)) with $k=2$ with XLM ([Conneau et al., 2019](https://aclanthology.org/2020.acl-main.747/)) as encoder:
```shell 
python3 run.py dep-bracket -k 2 -c configs/xlm.ini \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu \
    --output-folder results/dep-bracket-xlm
```

- Arc-Eager transition-based system ([Nivre and Fernández-González, 2002](https://aclanthology.org/J14-2002/)) where each state is represented with 1 position of the stack and 2 positions of the buffer. Use XLM ([Conneau et al., 2019](https://aclanthology.org/2020.acl-main.747/)) as encoder:
```shell 
python3 run.py dep-eager --stack 1 --buffer 2 -c configs/xlm.ini \
    train --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu \
    --output-folder results/dep-eager-xlm
```

- Hexa-Tagging ([Amini et al., 2023](https://aclanthology.org/2023.acl-short.124/)) with XLNet as encoder and the *head* pseudo-projective transformation (modes available are `head`, `head+path`, `path`) from [Nivre and Nilsson (2005)](https://aclanthology.org/P05-1013/):
```shell 
python3 run.py dep-hexa -c configs/xlnet.ini --proj head \
    --train treebanks/english-ewt/train.conllu \
    --dev treebanks/english-ewt/dev.conllu \
    --test treebanks/english-ewt/test.conllu \
    --output-folder -p results/dep-hexa-xlnet
```



## Evaluation 
Evaluate now the trained parser at `results/dep-bracket-xlm/parser.pt` with the same test file:
```shell 
python3 run.py dep-bracket --load results/dep-bracket-xlm/parser.pt \
    eval treebanks/english-ewt/test.conllu --batch-size 50
```

## Prediction 
Predict with the trained parser at `results/dep-hexa-xlnet`:
```shell 
python3 run.py dep-hexa --load results/dep-hexa-xlnet/parser.pt predict \
     treebanks/english-ewt/test.conllu results/dep-hexa-xlnet/pred.conllu --batch-size 50
```
