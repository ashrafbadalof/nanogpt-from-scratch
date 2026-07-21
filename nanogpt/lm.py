import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 8
batch_size = 16

with open("data/tinyshakespeare.txt", encoding='utf-8') as f:
    data = f.read()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab = sorted(set(list(data)))
vocab_size = len(vocab)

char_to_int = {value:index for index, value in enumerate(vocab)}
int_to_char = {index:value for index, value in enumerate(vocab)}

def encode(text):
    return [char_to_int[i] for i in text]

def decode(tokens):
    return ''.join([int_to_char[i] for i in tokens])
    
encoded_data = encode(data)
n_split = int(0.9* len(encoded_data))

train = torch.tensor(encoded_data[:n_split], dtype=torch.long)
val = torch.tensor(encoded_data[n_split:], dtype=torch.long)


def get_batch(data):
    n = torch.randint(0, len(data) - block_size,(batch_size,))
    x = torch.stack([data[i: i + block_size] for i in n])
    y = torch.stack([data[i+1: i+block_size +1] for i in n])
    x, y = x.to(device), y.to(device)
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self,idx, targets = None):
        logits = self.embedding_table(idx)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
model = BigramLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

max_steps = 1000
for step in range(max_steps):
    xb, yb = get_batch(train)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(loss)
print(f'final loss: {loss}')

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))