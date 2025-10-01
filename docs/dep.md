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
| `dep-hier` | Hierarchical Bracketing | [Ezquerro et al., (2025)](https://aclanthology.org/2025.acl-long.903/) | `variant`$\in$ `{proj, head, head+path, path, nonp}`| `proj` | 
| `dep-eager` | Arc-Eager system | [Nivre and Fernández-González (2002)](https://aclanthology.org/J14-2002/) | `stack`$\in\mathbb{N}$, `buffer`$\in\mathbb{N}$, `proj`$\in$`{None, head, head+path, path}` | `1`, `1`, `None`| w
| `dep-biaffine` | Biaffine dependency parser | [Dozat & Manning (2017)](https://arxiv.org/abs/1611.01734) | |

## Training 


- Absolute indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) with XLNet ([Yang et al., 2019](http://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)) as encoder. Add `--rel` argument as `<specific-args>` to exchange absolute for relative positions.

```shell 
python3 run.py dep-idx -c configs/xlnet.ini -d 0\
    train --train $DATA_FOLDER/english-ewt/train.conllu \
    --dev $DATA_FOLDER/english-ewt/dev.conllu \
    --test $DATA_FOLDER/english-ewt/test.conllu \
    --output-folder $RESULTS_FOLDER/dep-abs-xlnet
```

Or for multi-GPU training:
```shell 
CUDA_VISIBLE_DEVICES="0,2,3" torchrun --nproc_per_node=3 run.py dep-idx -c configs/xlnet.ini \
    train --train $DATA_FOLDER/english-ewt/train.conllu \
    --dev $DATA_FOLDER/english-ewt/dev.conllu \
    --test $DATA_FOLDER/english-ewt/test.conllu \
    --output-folder $RESULTS_FOLDER/dep-abs-xlnet
```


- PoS-based relative indexing ([Strzyz et al., 2019](https://aclanthology.org/N19-1077/)) with BiLSTMs as encoder.
```shell 
python3 run.py dep-pos -c configs/bilstm.ini -d 0 \
    train --train $DATA_FOLDER/english-ewt/train.conllu \
    --dev $DATA_FOLDER/english-ewt/dev.conllu \
    --test $DATA_FOLDER/english-ewt/test.conllu \
    --output-folder $RESULTS_FOLDER/dep-pos-bilstm 
```


- Bracketing encoding ([Strzyz et al., 2020](https://aclanthology.org/2020.coling-main.223/)) with $k=2$ with XLM ([Conneau et al., 2019](https://aclanthology.org/2020.acl-main.747/)) as encoder:
```shell 
python3 run.py dep-bracket -k 2 -c configs/xlm.ini \
    train --train $DATA_FOLDER/english-ewt/train.conllu \
    --dev $DATA_FOLDER/english-ewt/dev.conllu \
    --test $DATA_FOLDER/english-ewt/test.conllu \
    --output-folder $RESULTS_FOLDER/dep-bracket-xlm
```

- Arc-Eager transition-based system ([Nivre and Fernández-González, 2002](https://aclanthology.org/J14-2002/)) where each state is represented with 1 position of the stack and 2 positions of the buffer. Use XLM ([Conneau et al., 2019](https://aclanthology.org/2020.acl-main.747/)) as encoder:
```shell 
python3 run.py dep-eager --stack 1 --buffer 2 -c configs/xlm.ini \
    train --train $DATA_FOLDER/english-ewt/train.conllu \
    --dev $DATA_FOLDER/english-ewt/dev.conllu \
    --test $DATA_FOLDER/english-ewt/test.conllu \
    --output-folder $RESULTS_FOLDER/dep-eager-xlm
```

- Hexa-Tagging ([Amini et al., 2023](https://aclanthology.org/2023.acl-short.124/)) with XLNet as encoder and the *head* pseudo-projective transformation (modes available are `head`, `head+path`, `path`) from [Nivre and Nilsson (2005)](https://aclanthology.org/P05-1013/):
```shell 
python3 run.py dep-hexa -c configs/xlnet.ini --proj head \
    --train $DATA_FOLDER/english-ewt/train.conllu \
    --dev $DATA_FOLDER/english-ewt/dev.conllu \
    --test $DATA_FOLDER/english-ewt/test.conllu \
    --output-folder -p $RESULTS_FOLDER/dep-hexa-xlnet
```


## Evaluation 
Evaluate now the trained parser at `$RESULTS_FOLDER/dep-bracket-xlm/parser.pt` with the same test file:
```shell 
python3 run.py dep-bracket --load $RESULTS_FOLDER/dep-bracket-xlm/parser.pt \
    eval $DATA_FOLDER/english-ewt/test.conllu --batch-size 50
```

## Prediction 
Predict with the trained parser at `$RESULTS_FOLDER/dep-hexa-xlnet`:
```shell 
python3 run.py dep-hexa --load $RESULTS_FOLDER/dep-hexa-xlnet/parser.pt predict \
     $DATA_FOLDER/english-ewt/test.conllu $RESULTS_FOLDER/dep-hexa-xlnet/pred.conllu --batch-size 50
```


## Reproducibility 

### *Hierarchical Bracketing Encodings for Dependency Parsing as Tagging* ([Ezquerro et al., 2025a](https://aclanthology.org/2025.acl-long.903/))

[Ezquerro et al. (2025a)](https://aclanthology.org/2025.acl-long.903/) proposed hierarchical bracketing encodings for dependency trees and evaluated its different variants on the PTB ([Marcus et al., 1993](https://aclanthology.org/J93-2004/)) and CTB ([Xue et al., 2002](https://aclanthology.org/C02-1145/)) with dependency annotations ([Marneffe and Manning, 2008](https://aclanthology.org/W08-1301/)) and [Universal Dependencies](https://universaldependencies.org/). 


In their experiments, they compared 4-bit and 7-bit encoding ([Gómez-Rodríguez et al., 2023](https://aclanthology.org/2023.emnlp-main.393)), and hexatagging ([Amini et al., 2023](https://aclanthology.org/2023.acl-short.124/)) with different levels of projectivity.

To reproduce their results with the 4-bit encoding, this implementation uses the argument `--proj` to specify whether a pseudo-projective transformtion ([Nivre and Nilsson, 2005](https://aclanthology.org/P05-1013/)) should be applied before encoding (and the reverse process after decoding). If not specified, no pseudo-projectivity is applied. The hexa-tagging by default applies the `head` transformation. 

The hierarchical bracketing encoding (`dep-hier`) accepts the parameter `--variant` to specify which projectivity level is supported by the algorithm:
- If `proj`, no pseudo-projectivity transformation is applied and the encoding tries to transform dependency trees that might not be projective.
- If `nonp`, the algorithm uses the non-projective encoding (which uses subindices to skip matching symbols in the decoding process).
- Otherwise, specifies the pseuo-projective transformation from [Nivre and Nilsson (2005)](https://aclanthology.org/P05-1013/): `head`, `head+path` or `path`.

For example, to train the hierarchical bracketing encoding with the non-projective variant on the English-EWT dataset:

```shell
python3 run.py dep-hier --variant nonp -c configs/xlnet.ini \
    train --train $DATA_FOLDER/english-ewt/train.conllu \
          --dev $DATA_FOLDER/english-ewt/dev.conllu \
          --test $DATA_FOLDER/english-ewt/test.conllu \
          --output-folder $RESULTS_FOLDER/dep-hier
```


### *Bringing Emerging Architectures to Sequence Labeling in NLP* ([Ezquerro et al., 2025c](https://arxiv.org/abs/2509.25918))

[Ezquerro et al. (2025c)](https://arxiv.org/abs/2509.25918) conducted experiments on multiple dependency encodings using a diffusion and adversarial tagger. 

> [!NOTE]  
> For the diffusion model, note that the parameters of `[dec_conf]` are loaded from the configuration file, while for the adversarial tagger, the `[dis_conf]` is used. To call the diffusion (`dit`) or adversarial (`gat`) model, use the `dit` and `gat` variants of each graph encoding (introducing the flag between the `grp` prefix and encoding suffix). 

An example to run hexatagging on the English-EWT with the diffusion tagger with the pseudo-projective transformation:
```shell 
python3 run.py dep-dit-hexa -c configs/xlnet.ini \
    train --train $DATA_FOLDER/english-ewt/train.conllu \
          --dev $DATA_FOLDER/english-ewt/dev.conllu \
          --test $DATA_FOLDER/english-ewt/test.conllu \
          --output-folder $RESULTS_FOLDER/dep-dit-hexa
```

