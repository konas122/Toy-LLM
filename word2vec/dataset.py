import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader

from vocab import *


class Word2VecDataset(Dataset):
    def __init__(self, 
                 window_size: int = 5,
                 neg_sample_num: int = 5,
                 file_path='../data/news/word2vec_train_words.txt'):
        self.window_size = window_size
        self.neg_sample_num = neg_sample_num

        self.vocab = []
        self.sentences = []

        self._build_vocab(file_path)
        self.train_pairs = self._create_training_pairs()

    def _build_vocab(self, file_path):
        vocab = set()
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = re.findall(r'\w+', line.lower())
                    if tokens:
                        self.sentences.append(tokens)
                        vocab.update(tokens)

            self.vocab = list(vocab)
            self.null_idx = len(self.vocab)
            self.vocab.append('\u3000')
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        else:
            raise RuntimeError("{} not exists!!\n", file_path)

    def _create_training_pairs(self) -> List[Tuple[int, List[int]]]:
        pairs = []
        for sentence in self.sentences:
            word_ids = [self.word2idx[word] for word in sentence 
                       if word in self.word2idx]

            for i, center_word_idx in enumerate(word_ids):
                context_indices = []
                for j in range(-self.window_size, self.window_size + 1):
                    if j == 0:
                        continue
                    context_pos = i + j
                    if 0 <= context_pos < len(word_ids):
                        context_indices.append(word_ids[context_pos])

                if context_indices:
                    pairs.append((center_word_idx, context_indices))

        return pairs

    def get_vocab(self):
        return Vocab(self.vocab, self.word2idx, self.idx2word)

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        center_word, context_words = self.train_pairs[idx]

        max_context = 2 * self.window_size
        if len(context_words) > max_context:
            context_words = np.random.choice(context_words, max_context, replace=False)
        else:
            context_words = context_words + [self.null_idx] * (max_context - len(context_words))

        neg_word_ids = []
        for _ in range(max_context):
            neg_samples = []
            while len(neg_samples) < self.neg_sample_num:
                neg_idx = np.random.randint(0, len(self.vocab))
                if neg_idx != center_word and neg_idx not in context_words:
                    neg_samples.append(neg_idx)
            neg_word_ids.extend(neg_samples)

        context_and_neg = context_words + neg_word_ids

        labels = []
        for word_idx in context_words:
            labels.append(1.0 if word_idx != self.null_idx else 0.0)
        labels.extend([0.0] * len(neg_word_ids))

        return (torch.LongTensor([center_word]),
                torch.LongTensor(context_and_neg),
                torch.FloatTensor(labels))

        # neg_word_ids = []
        # for _ in range(len(context_words)):
        #     # neg_words = np.random.choice(self.vocab, size=self.neg_sample_num)
        #     # neg_word_ids = [self.word2idx[word] for word in neg_words 
        #     #     if word in self.word2idx and self.word2idx[word] not in context_words]
        #     neg_samples = []
        #     while len(neg_samples) < self.neg_sample_num:
        #         neg_idx = np.random.choice(self.vocab, size=1)
        #         if neg_idx != center_word and neg_idx not in context_words:
        #             neg_samples.append(neg_idx)
        #     neg_word_ids.extend(neg_samples)

        # context_and_neg = context_words + neg_word_ids

        # labels = ([1.] * len(context_words) + 
        #          [0.] * (len(neg_word_ids)))

        # return (torch.LongTensor([center_word]),
        #         torch.LongTensor(context_and_neg),
        #         torch.FloatTensor(labels))

def create_data_loader(
                      batch_size: int = 32,
                      window_size: int = 5,
                      neg_sample_num: int = 5) -> Tuple[DataLoader, Dict[str, int]]:
    dataset = Word2VecDataset(
        window_size=window_size,
        neg_sample_num=neg_sample_num
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return data_loader, dataset.vocab, dataset.get_vocab()



if __name__ == "__main__":
    batch_size = 64

    data_loader, vocab, _ = create_data_loader(
        batch_size=batch_size,
        window_size=5,
        neg_sample_num=5
    )

    for data in data_loader:
        print(data)
