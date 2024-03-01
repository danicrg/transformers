import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm

torch.manual_seed(1337)


class Tokenizer:
    """
    Simple character-wise tokenizer generated from a text
    """

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return [self.stoi.get(c, -1) for c in s]

    def decode(self, l):
        return "".join([self.itos.get(i, "UNK") for i in l])


def read_data():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embed, config.n_embed // config.n_head, bias=False)
        self.query = nn.Linear(config.n_embed, config.n_embed // config.n_head, bias=False)
        self.value = nn.Linear(config.n_embed, config.n_embed // config.n_head, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )  # So these are not weights of the model

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out


class BatchMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.key = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=False)

        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.att_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, nh, T, T)
        wei = self.dropout(wei)

        out = wei @ v  # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        
        out = self.resid_dropout(out)

        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        self.sa = BatchMultiHeadAttention(config)
        self.ffw = FeedFoward(config)

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connections
        x = x + self.ffw(self.ln2(x))  # Residual connections
        return x


class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embed)
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        idx = idx[:, -self.block_size :]
        B, T = idx.shape

        tok_embed = self.token_embedding_table(idx)  # (B, T, n_embed) C being embed size in this case
        pos_embed = self.position_embedding(torch.arange(T, device=device))  # (T, C)
        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size) C being vocab size in this case

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # Shape is (B, T, C)

            last_logits = logits[:, -1, :]  # Becomes (B, C)

            probs = F.softmax(last_logits, dim=-1)  # (B, C)

            index = torch.multinomial(probs, num_samples=1)  # Becomes (B, 1)

            idx = torch.cat((idx, index), dim=-1)  # Becomes (B, T + 1)

        return idx


@torch.no_grad()
def estimate_loss(model: nn.Module, n_estimations=100):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(n_estimations)
        for k in range(n_estimations):
            x, y = get_batch(split)
            _, loss = model(x.to(device), y.to(device))
            losses[k] = loss

        out[split] = losses.mean()
    model.train()
    return out


device = "mps" if torch.backends.mps.is_available() else "cpu"


# Training
text = read_data()
tokenizer = Tokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.int64)

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split, batch_size=4, block_size=8):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class Config:
    vocab_size = tokenizer.vocab_size
    n_steps = 5_000
    batch_size = 64
    n_embed = 384
    block_size = 256
    dropout = 0.2
    n_layer = 6
    n_head = 6
    learning_rate = 4e-4


config = Config()

m = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)

print("Number of parameters: ", sum(p.numel() for p in m.parameters() if p.requires_grad))

for i in tqdm(range(config.n_steps)):

    xb, yb = get_batch("train", batch_size=config.batch_size, block_size=config.block_size)
    _, loss = m(xb.to(device), yb.to(device))

    if i % 100 == 0:
        print(estimate_loss(m))
        context = get_batch("val", batch_size=1, block_size=1)[0]
        print("Writing Sample:")
        print(tokenizer.decode(m.generate(idx=context.to(device), max_new_tokens=100)[0].tolist()))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

m.eval()
