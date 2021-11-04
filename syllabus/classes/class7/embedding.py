import torch
from torch import nn
from typing import Tuple
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def gensim_to_torch_embedding(gensim_wv: KeyedVectors, add_padding = True,
    add_unknown = True) -> Tuple[torch.Tensor, dict]: 
    """
    A function that makes gensim (numpy) word embeddings into torch (tensor) word embeddings.
    Takes in word embedding vectors and returns a tuple containing tensor word embedding
    vectors and a dictionary with words as keys and index as values.
    Args:
        add_padding: Defaults to True.
        add_unknown: Defaults to True.
    """
    embedding_size = gensim_wv.vectors.shape[1]

    # word embedding for unknown words: defined as the mean of all word embeddings
    unk_emb = np.mean(gensim_wv.vectors, axis=0).reshape((1, embedding_size))
    # vector of 0s with embedding size for padding sentences 
    pad_emb = np.zeros((1, gensim_wv.vectors.shape[1]))

    # add the new embedding: combining everything
    embeddings = np.vstack([gensim_wv.vectors, unk_emb, pad_emb])

    # converting to torch
    weights = torch.FloatTensor(embeddings)

    emb_layer = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=-1)

    # creating vocabulary
    vocab = gensim_wv.key_to_index

    # adding arguments
    if add_unknown == True:
        vocab["UNK"] = weights.shape[0] - 2
    else:
        pass
    if add_padding == True:
        vocab["PAD"] = emb_layer.padding_idx
    else:
        pass

    return emb_layer, vocab