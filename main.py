import torch
import tiktoken
from utils import train_model
from gpt2.gpt_model import GPTModel
from gpt2.config import GPT_CONFIG_124M
from gpt2.util import download_and_load_gpt2, load_weights
from dataset import create_dataloader, prepare_spam_data, SpamDataset, prepare_instruction_data


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "model_size": "124M"},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "model_size": "355M"},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "model_size": "774M"},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "model_size": "1558M"},
}

model_name = "gpt2-medium (355M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])


file_path = "data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=NEW_CONFIG["context_length"],
    stride=NEW_CONFIG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=NEW_CONFIG["context_length"],
    stride=NEW_CONFIG["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

model = GPTModel(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")

settings, params = download_and_load_gpt2(model_size=NEW_CONFIG["model_size"], models_dir="draft/gpt2")
load_weights(model, params)

num_epochs = 1
model.to(device)
train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

model.eval()
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     }, 
#     "model_and_optimizer.pth"
# )

prepare_spam_data()
train_dataset = SpamDataset(
    csv_file="data/train.csv",
    max_length=None,
    tokenizer=tokenizer
)

print(train_dataset.max_length)

prepare_instruction_data()
