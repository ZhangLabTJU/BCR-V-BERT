# BCR-V-BERT: Antibody language model based on BERT fused V gene

## Overview

Official repository for BCR-V-BERT, an antibody-specific transformer language model pre-trained separately on 372M antibody heavy sequences and 3M antibody light sequences. The model can embed the CDR3 or CDR sequences of light and heavy chain antibodies.

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see License file).
This software is not to be used for commercial purposes.

## Setup

### Environment

To download our code, we recommend creating a clean conda environment with python v3.9, and you will also need to install PyTorch v2.0.0.
```bash
conda create --name public-env python=3.9
conda activate public-env
```

To use BCR-V-BERT, you can clone this repository and install the package locally:
```bash
$ git clone https://user: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@github.com/ZhangLabTJU/BCR-V-BERT.git
$ cd BCR-V-BERT/
$ pip install -r requirements.txt
$ python setup.py install
```
### Pre-trained models

The pre-trained model for this project is hosted on Hugging Face at [xqh/BCR-V-BERT](https://huggingface.co/xqh/BCR-V-BERT). When you follow the standard installation process, the system will automatically clone the pre-trained model from Hugging Face into the BCR_V_BERT/model_pretrained directory.

During the installation process by using setup.py, the installation script will automatically handle the downloading and configuration of the pre-trained model. If everything proceeds smoothly, no additional actions are required from your side.

If the automatic cloning step fails to complete successfully, please manually clone the pre-trained model to the designated directory:

```bash
$ git clone https://huggingface.co/xqh/vbert BCR_V_BERT/model_pretrained
```

Verify that the files have been successfully downloaded to the BCR_V_BERT/model_pretrained directory.


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
```

## Citing this work
If you use BCR-V-BERT in your research, please cite our work as follows:

[Add citation details here]
