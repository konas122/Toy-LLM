"""
GPT2-small (the 124M configuration we already implemented):
    "emb_dim" = 768
    "n_layers" = 12
    "n_heads" = 12

GPT2-medium:
    "emb_dim" = 1024
    "n_layers" = 24
    "n_heads" = 16

GPT2-large:
    "emb_dim" = 1280
    "n_layers" = 36
    "n_heads" = 20

GPT2-XL:
    "emb_dim" = 1600
    "n_layers" = 48
    "n_heads" = 25
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True        # Query-Key-Value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "model_size": "124M"},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "model_size": "355M"},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "model_size": "774M"},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "model_size": "1558M"},
}
