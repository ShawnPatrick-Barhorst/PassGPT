import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
import pandas as pd

import tqdm
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'



class TransformerBlock(nn.Module):
  def __init__(self, model_dim, num_heads):
    super().__init__()
    self.attention = nn.MultiheadAttention(model_dim, num_heads)
    self.layer_norm1 = nn.LayerNorm(model_dim)
    self.linear = nn.Linear(model_dim, model_dim)
    self.layer_norm2 = nn.LayerNorm(model_dim)

  def forward(self, x, mask=None, padding_mask=None):
    attn_output, _ = self.attention(x, x, x, attn_mask=mask, key_padding_mask = padding_mask)
    x = self.layer_norm1(x + attn_output)
    linear_output = self.linear(x)
    x = self.layer_norm2(x + linear_output)
    return x

class Encoder(nn.Module):
  def __init__(self, vocab_size, block_size, embed_size, padding_id):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_id)
    self.pos_encoding = nn.Embedding(block_size, embed_size)

  def forward(self, x):
    token_embedding = self.embedding(x) # (batch_size, sequence_length, embed_size)
    pos_indices = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    pos_encoding = self.pos_encoding(pos_indices).expand(x.size(0), -1, -1)
    x = token_embedding + pos_encoding
    x = x.transpose(0, 1) # (sequence_length, batch_size, embed_size)
    return x

class GPT(nn.Module):
  def __init__(self, vocab_size, block_size, embed_size, model_dim, num_heads, padding_id):
    super().__init__()
    self.encoder = Encoder(vocab_size, block_size, embed_size, padding_id)
    self.transformer_blocks = nn.ModuleList(
        [TransformerBlock(model_dim, num_heads) for _ in range(3)]
    )
    self.linear = nn.Linear(model_dim, vocab_size)

  def forward(self, x, padding_mask=None):
    x = self.encoder(x) # (sequence_length, batch_size, embed_size)

    seq_len = x.size(0)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #mask = mask==1


    for block in self.transformer_blocks:
      x = block(x, mask, padding_mask)

    x = x.transpose(0, 1) # (batch_size, sequence_length, model_dim)
    x = self.linear(x) # (batch_size, sequence_length, vocab_size)
    return x
    
    

"""##Data Wrangling"""

train = pd.read_csv('PassGPT/train.csv')
train = train.sample(frac=1).reset_index(drop=True)
train = train

test = pd.read_csv('PassGPT/test.csv')
test = test.sample(frac=1).reset_index(drop=True)
test = test

text_content = "\n".join(train['Text'].astype(str))

with open('data.txt', 'w') as file:
    file.write(text_content)

max_length = train['Text'].str.len().max()
max_length = 20

from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Initialize a BPE tokenizer
tokenizer = ByteLevelBPETokenizer()

tokenizer.pre_tokenizer = Whitespace()
# Customize training
tokenizer.train(
    files=["data.txt"],  # List of file paths
    vocab_size=30000,  # Set the vocabulary size
    min_frequency=2,  # Set the minimum frequency of tokens to keep
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]  # Define special tokens
)

tokenizer.enable_padding(length=max_length)
tokenizer.enable_truncation(max_length=max_length)

bos_token = "<s>"
eos_token = "</s>"

tokenizer.post_processor = TemplateProcessing(
    single=f"{bos_token} $A {eos_token}",
    pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1",
    special_tokens=[
        (bos_token, tokenizer.token_to_id(bos_token)),
        (eos_token, tokenizer.token_to_id(eos_token)),
    ]
)

train = train['Text'].tolist()
validation = test['Text'].tolist()



class PasswordDataset(torch.utils.data.Dataset):
  def __init__(self, passwords, tokenizer):
    self.tokenized_passwords = tokenizer.encode_batch(passwords)

  def __len__(self):
    return len(self.tokenized_passwords)

  def __getitem__(self, idx):
    return torch.tensor(self.tokenized_passwords[idx].ids), torch.tensor(self.tokenized_passwords[idx].attention_mask, dtype=bool)

train_dataset = PasswordDataset(train, tokenizer)
validation_dataset = PasswordDataset(validation, tokenizer)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=False)

def train_model(model, trainloader, validationloader, epochs, device, vocab_size, padding_id):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()  # Set model to training mode

    for epoch in range(1, epochs + 1):
        total_loss = 0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}', unit='batch', ncols=100)

        for batch in pbar:
            input_ids, padding_mask = batch
            input_ids, padding_mask = input_ids.to(device), padding_mask.to(device)

            padding_mask = (padding_mask == 0)

            optimizer.zero_grad()

            logits = model(input_ids, padding_mask=padding_mask)

            logits = logits.view(-1, vocab_size)
            input_ids = input_ids.view(-1)

            loss = F.cross_entropy(logits, input_ids, ignore_index=padding_id)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)


        avg_loss = total_loss / len(trainloader.dataset)


        #validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
          for input_ids, padding_mask in validationloader:
            input_ids, padding_mask = input_ids.to(device), padding_mask.to(device)

            padding_mask = (padding_mask == 0)

            logits = model(input_ids, padding_mask=padding_mask)
            logits = logits.view(-1, vocab_size)

            input_ids = input_ids.view(-1)

            val_loss += F.cross_entropy(logits, input_ids, ignore_index=padding_id).item() * input_ids.size(0)

        val_loss /= len(validationloader.dataset)


        print(f'Epoch {epoch}:\t Training Loss: {avg_loss:.6f}\tValidation Loss: {val_loss:.6f}')

model_dim = 256
num_heads = 4
padding_id = 0
embed_size = 256
vocab_size = 30000
block_size = max_length
batch_size = 1

model = GPT(vocab_size, block_size, embed_size, model_dim, num_heads, padding_id)
model = model.to(device)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.zeros_(m.bias)

model.apply(init_weights)


train_model(model, trainloader, validationloader, 1, device, vocab_size, 0)

def generate(model, tokenizer, max_length=max_length):
  model.eval()

  idx = torch.tensor([[tokenizer.token_to_id("<s>")]], dtype=torch.long, device=device)

  for _ in range(max_length):
    block_size = idx.size(1)
    logits = model(idx)
    logits = logits[:, -1, :]

    probs = F.softmax(logits, dim=-1)

    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)

    if idx_next == tokenizer.token_to_id("</s>"):
      break
  return idx

for _ in range(10):
  out = generate(model, tokenizer)
  print(tokenizer.decode(out[0].tolist()))

torch.save(model.state_dict(), 'model.pt')
