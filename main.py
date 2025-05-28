import torch
import tiktoken
from utils import train_model
from functools import partial
from gpt2.gpt_model import GPTModel
from gpt2.config import GPT_CONFIG_124M
from torch.utils.data import DataLoader
from gpt2.util import download_and_load_gpt2, load_weights
from dataset import prepare_instruction_data, InstructionDataset, custom_collate_fn, format_input


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "model_size": "124M"},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "model_size": "355M"},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "model_size": "774M"},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "model_size": "1558M"},
}

model_name = "gpt2-medium (355M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

model = GPTModel(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings, params = download_and_load_gpt2(model_size=NEW_CONFIG["model_size"], models_dir="draft/gpt2")
load_weights(model, params)

num_epochs = 2
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

data = prepare_instruction_data()

train_portion = int(len(data) * 0.85)                   # 85% for training
test_portion = int(len(data) * 0.1)                     # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

batch_size = 8
num_workers = 0

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)
