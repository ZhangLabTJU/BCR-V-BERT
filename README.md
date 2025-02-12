# BCR-V-BERT: Antibody language model based on BERT fused V gene

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see License file).
This software is not to be used for commerical purposes.

Commercial users/for profit organisations can obtain a license from Cambridge Enterprise.

## Overview

Official repository for BCR-V-BERT, an antibody-specific transformer language model pre-trained separately on 372M antibody heavy sequences and 3M antibody light sequences. The model can embed the CDR3 or CDR sequences of light and heavy chain antibodies.

## Setup

### Environment

To download our code, we recommend creating a clean conda environment with python v3.9, and you will also need to install PyTorch v2.0.0.

To use BCR-V-BERT, you can clone this repository and install the package locally:
```bash
$ git clone git@github.com/ZhangLabTJU/BCR-V_BERT/main
$ python setup.py install
```

## Available models

| Model | Dataset | Epoch | Description |
|-------|------------------------------------------------------------|------|---------------|
| cdrh  | 372,028,240 antibody cdrh1,2,3 sequences from OAS database | 0.15 | Heavy chain cdr pretrained model |
| cdrh3 | 372,028,240 antibody cdrh3 sequences from OAS database     | 0.15 | Heavy chain cdr3 pretrained model |
| cdrl  | 3,705,441 antibody cdrl1,2,3 sequences from OAS database   | 20   | Light chain cdr pretrained model |
| cdrl3 | 3,705,441 antibody cdrl3 from OAS database                 | 20   | Light chain cdr3 pretrained model |

## Usage

BCR-V-BERT can be used in different ways and for a variety of usecases.
    
- Embeddings: Generates sequence embeddings. The output is a list of embedding tensors, where each tensor is the embedding for the corresponding sequence. Each embedding has dimension `[(Length + 2) x 768]`.
    
```python
from BCR_V_BERT import BCR_V_BERT_Runner

cdrh_model = BCR_V_BERT_Runner(model = 'cdrh')
cdrh3_model = BCR_V_BERT_Runner(model = 'cdrh3')

cdrh1_seq = 'GFTISDYW'
cdrh2_seq = 'ITPAGGYT'
cdrh3_seq = 'ARFVFFLPYAMDY'
vgene = 'IGHV1-52'

# Embeddings
cdrh_embeddings = cdrh_model.embed([cdrh1_seq+'|'+cdrh2_seq+'|'+cdrh3_seq],[vgene])
cdrh3_embeddings = cdrh3_model.embed([cdrh3_seq],[vgene])

## Citing this work
