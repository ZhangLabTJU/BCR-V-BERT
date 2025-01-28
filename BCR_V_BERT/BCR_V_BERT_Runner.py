import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import BCR_V_BERT
project_path = os.path.dirname(os.path.realpath(BCR_V_BERT.__file__))
# print("project_path Directory:", project_path)

path_VBERT = project_path
path_utils = os.path.join(path_VBERT, 'utils')
path_vbert = os.path.join(path_VBERT, 'model')
sys.path.append(path_VBERT)
sys.path.append(path_utils)
sys.path.append(path_vbert)

from featurization import get_aa_bert_tokenizer
import toolkit as tk

class BCR_V_BERT_Runner():
    def __init__(self,model):
        """
        Initialize bcr-v-bert model.

        Args:
            path_model: bcr-v-bert model, including cdrh, cdrl

        Returns:
        """
        self.path_model = os.path.join(path_VBERT,'model_pretrained',model)

    def embed(self, sequences, vgenes, hidden_layer=-1):
        """
        Embed a list of sequences.

        Args:
            sequences (list): list of sequences
            vgenes (list): list of vgenes
            hidden_layer (int): which hidden layer to use (0 to 12)

        Returns:
            list(torch.Tensor): list of embeddings (one tensor per sequence)

        """
        max_len = max(len(sequence) for sequence in sequences)
        max_len_rounded = int(tk.min_power_greater_than(max_len+2, base=2))
        tok = get_aa_bert_tokenizer(max_len_rounded)

        v_vocab=np.load(os.path.join(self.path_model,'v_vocab.npy'),allow_pickle=True)
        v_vocab = v_vocab.tolist()
        vgene_tensors = [v_vocab.index(vgene) for vgene in vgenes]
        embeddings = tk.get_transformer_embeddings(
            model_dir=self.path_model,
            tok=tok,
            seqs=sequences,
            vgene=vgene_tensors,
            layers=[hidden_layer],
            method="mean",
            device=0,
            max_len=max_len_rounded
        )
        return embeddings