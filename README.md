# nanogpt-from-scratch

A small decoder-only transformer language model trained on tinyshakespeare, character level, written from scratch in PyTorch.

I built each piece as its own exercise first: attention, multi-head attention, LayerNorm,
the transformer block and then assembled them here. No tutorial code copied.

**Setup:** 6 blocks, 6 heads, 384 dims, 256-token context, ~10M params.

**Results:** 1.2 train loss after 5000 steps on a Colab T4. Output in `sample.txt`.

```bash
python train.py       # saves shakespeare_gpt.pt
python generate.py    # writes sample.txt
```

Training wants a real GPU. Generation runs fine on 4GB.
