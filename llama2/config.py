import torch


LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,    # Vocabulary size
    "context_length": 4096, # Context length
    "emb_dim": 4096,        # Embedding dimension
    "n_heads": 32,          # Number of attention heads
    "n_layers": 32,         # Number of layers
    "hidden_dim": 11008,    # Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16 # Lower-precision dtype to reduce memory usage
}
