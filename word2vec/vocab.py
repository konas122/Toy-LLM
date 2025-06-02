from typing import List, Dict, Union

class Vocab:
    def __init__(self,
                 vocab: List[str],
                 word2idx: Dict[str, int],
                 idx2word: Dict[int, str]):
        self.vocab: List[str] = vocab
        self.word2idx: Dict[str, int] = word2idx
        self.idx2word: Dict[int, str] = idx2word

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx: Union[int, float]) -> str:
        return self.idx2word[int(idx)]

    def __getitem__(self, word: str) -> int:
        return self.word2idx[word]
