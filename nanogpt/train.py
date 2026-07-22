import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 256
batch_size = 64
n_embed = 384
n_heads = 6
n_layers = 6
max_steps = 5000
lr = 3e-4
dropout = 0.2
eval_iters = 200 # how many batches to average over
eval_interval = 500

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

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_heads, dropout):
        super().__init__()
        self.q_proj = nn.Linear(n_embed, n_embed)
        self.k_proj = nn.Linear(n_embed, n_embed)
        self.v_proj = nn.Linear(n_embed, n_embed)
        self.o_proj = nn.Linear(n_embed, n_embed)
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, T, _ = x.shape
        q = q.reshape((B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        k = k.reshape((B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        v = v.reshape((B, T, self.n_heads, self.head_dim)).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attention_scores = attention_scores.masked_fill(mask==0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ v
        output = output.transpose(1, 2).contiguous().view(B, T, self.n_embed)
        out = self.o_proj(output)
        return out

class MLP(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embed, n_embed*4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(n_embed*4, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc2(self.gelu(self.fc1(x)))
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, n_embed, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, correction=0)
        normalized = (x - mean) / (var + self.eps)**0.5
        scaled = normalized * self.gamma + self.beta
        return scaled

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads, dropout):
        super().__init__()
        self.ln1 = LayerNorm(n_embed)
        self.mha = MultiHeadAttention(n_embed, n_heads, dropout)
        self.ln2 = LayerNorm(n_embed)
        self.mlp = MLP(n_embed, dropout)
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split_name, split_data in [('train', train), ('val', val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split_data)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.ln = LayerNorm(n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        block = self.blocks(x)
        ln_final = self.ln(block)
        logits = self.lm_head(ln_final)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
model = GPTLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(max_steps):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch(train)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "shakespeare_gpt.pt")

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))