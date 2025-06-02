import torch
import torch.nn as nn
from typing import List

from vocab import *


class Doc2Vec(nn.Module):
    def __init__(self, word_embeddings: nn.Embedding, aggregation='mean'):
        super().__init__()

        self.word_embeddings = word_embeddings
        self.aggregation = aggregation

        self.word_embeddings.weight.requires_grad = False

    def forward(self, word_indices: torch.Tensor) -> torch.Tensor:
        word_embeds = self.word_embeddings(word_indices)    # [batch_size, seq_len, embed_dim]

        if self.aggregation == 'mean':
            mask = (word_indices != 0).float().unsqueeze(-1)
            doc_vector = (word_embeds * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.aggregation == 'max':
            doc_vector = torch.max(word_embeds, dim=1)[0]

        return doc_vector


def _tokenize_by_vocab(vocab: Vocab, text: str) -> List[str]:
    tokens = []
    start = 0
    text = text.strip()

    while start < len(text):
        max_len = 0
        matched_token = None

        for token in vocab.vocab:
            if text[start:].startswith(token) and len(token) > max_len:
                max_len = len(token)
                matched_token = token

        if matched_token:
            tokens.append(matched_token)
            start += len(matched_token)
        else:
            start += 1

    return tokens


def text_to_vector(model: Doc2Vec, vocab: Vocab, text: str) -> torch.Tensor:
    words = _tokenize_by_vocab(vocab, text.lower())
    word_indices = [vocab[word] for word in words]
    indices_tensor = torch.tensor([word_indices], dtype=torch.long)

    with torch.no_grad():
        vector = model(indices_tensor)

    return vector
