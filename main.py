from torch.optim import AdamW

from transformer import Transformer, get_lr

src_vocab_size = 1000
tgt_vocab_size = 1000
max_len = 500
warmup_steps = 4000
total_steps = 100,000

## Hyperparams
n_layer = 6
d_model = 512
d_ff = 2048
nhead = 8
p_dropout = 0.1
ls_eps = 0.1

betas = (0.9, 0.98)
eps = 1e-9

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    nhead=nhead,
    n_layer=n_layer,
    d_ff=d_ff,
    p_dropout=p_dropout,
    max_len=max_len
)
optimizer = AdamW(model.parameters(), lr=1.0, betas=betas, eps=eps)
scheduler = get_lr(optimizer, warmup_steps, d_model)
