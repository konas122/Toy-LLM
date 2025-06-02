import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import *
from dataset import create_data_loader
from doc2vec import Doc2Vec, text_to_vector


def skip_gram(center, contexts_and_neg, embed_v, embed_u):
    v: torch.Tensor = embed_v(center)
    u: torch.Tensor = embed_u(contexts_and_neg)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = F.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none"
        )
        return out.mean(dim=1)


def get_similar(vocab: Vocab, query_token, k, embed: nn.Embedding):
    W = embed.weight.data
    x = W[vocab.word2idx[query_token]]
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:
        print(f'sim={float(cos[i]):.3f}: {vocab.idx2word[i]}')


def trainer(net: nn.Module, data_loader, lr, num_epochs):
    loss = SigmoidLoss()
    optim = torch.optim.Adam(net.parameters(), lr = lr)

    for epoch in range(num_epochs):
        idx = 0
        for center_word, contexts_and_neg, labels in data_loader:
            idx += 1
            optim.zero_grad()
            pred = skip_gram(center_word, contexts_and_neg, net[0], net[1])
            l = loss(pred.reshape(labels.shape).float(), labels.float())
            l = l.sum()
            l.backward()
            optim.step()

            if idx % 900 == 0:
                print(f'epoch {epoch + 1} loss: {l / batch_size:.3f}')


if __name__ == "__main__":
    lr = 0.01
    num_epochs = 4
    batch_size = 32
    embed_size = 100

    data_loader, vocab, info = create_data_loader(
        batch_size=batch_size,
        window_size=5,
        neg_sample_num=5
    )

    net = nn.Sequential(
        nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
        nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
    )
    trainer(net, data_loader, lr, num_epochs)

    test_file_1 = "data/amnews.txt"
    test_file_2 = "data/plnews.txt"
    text_1 = text_2 = ''

    with open(test_file_1, 'r', encoding='utf-8') as f:
        text_1 = f.read()

    with open(test_file_2, 'r', encoding='utf-8') as f:
        text_2 = f.read()

    doc2vec = Doc2Vec(net[0])
    vec_1 = text_to_vector(doc2vec, info, text_1).numpy()
    vec_2 = text_to_vector(doc2vec, info, text_2).numpy()

    print(f"vec_1:\n{vec_1}\n")
    print(f"vec_2:\n{vec_2}")
