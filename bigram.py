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
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return [self.stoi.get(c, -1) for c in s]
    
    def decode(self, l):
        return ''.join([self.itos.get(i, "UNK") for i in l])


def read_data():
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text


text = read_data()
tokenizer = Tokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.int64)

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size=4, block_size=8):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx) # (B, T, C) C being vocab size in this case
        
        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        logits = logits.view(B, T, C)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
           logits, _ =  self(idx) # Shape is (B, T, C)

           last_logits = logits[:, -1, :] # Becomes (B, C)

           probs = F.softmax(last_logits, dim=-1) # (B, C)

           index = torch.multinomial(probs, num_samples=1) # Becomes (B, 1)

           idx = torch.cat((idx, index), dim=-1) # Becomes (B, T + 1)

        return idx
    
@torch.no_grad()
def estimate_loss(model: nn.Module, n_estimations=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(n_estimations)
        for k in range(n_estimations):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss

        out[split] = losses.mean()
    model.train()
    return out



n_steps = 10_000
batch_size = 32

m = BigramLanguageModel(tokenizer.vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for i in tqdm(range(n_steps)):

    xb, yb = get_batch("train", batch_size=batch_size, block_size=1)
    _, loss = m(xb, yb)

    if i % 100 == 0:
        print(estimate_loss(m))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

m.eval()
print(tokenizer.decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.int64), max_new_tokens=100)[0].tolist()))

