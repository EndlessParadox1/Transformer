import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader

PAD_id = 0
UNK_id = 1
SOS_id = 2
EOS_id = 3
MAX_LEN = 500

def get_lr(optimizer: Optimizer, warmup_steps: int, d_model: int):
    def lr_lambda(step: int):
        step += 1
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return LambdaLR(optimizer, lr_lambda)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, p_dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.P = torch.zeros((1, max_len, d_model))
        x = (torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
             / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x: Tensor) -> Tensor:
        x += self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, nhead: int, n_layer: int, d_ff: int, p_dropout: float, max_len: int):
        super().__init__()
        input_emb = nn.Embedding(src_vocab_size, d_model)
        target_emb = nn.Embedding(tgt_vocab_size, d_model)
        pos_emb = PositionalEncoding(d_model, p_dropout, max_len)
        self.input_emb = lambda x: pos_emb(input_emb(x) * d_model ** 0.5)
        self.target_emb = lambda x: pos_emb(target_emb(x) * d_model ** 0.5)

        self.cell = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            dim_feedforward=d_ff,
            dropout=p_dropout,
            activation='gelu',
            batch_first=True,
        )

        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.linear.weight = target_emb.weight

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        device = src.device
        src_padding_mask = (src == PAD_id).to(device)
        tgt_padding_mask = (tgt == PAD_id).to(device)
        tgt_mask = self.cell.generate_square_subsequent_mask(tgt.size(1)).to(device)

        src = self.input_emb(src)
        tgt = self.target_emb(tgt)

        output = self.cell(
            src=src,
            src_key_padding_mask=src_padding_mask,
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        ) # [B, T, dm]
        output = self.linear(output) # [B, T, tgt_vocab_size]
        return output

    def encode(self, src: Tensor) -> Tensor:
        device = src.device
        src_padding_mask = (src == PAD_id).to(device)
        src = self.input_emb(src)
        output = self.encoder(
            src=src,
            src_key_padding_mask=src_padding_mask,
        )
        return output

    def decode(self, tgt: Tensor, memory: Tensor) -> Tensor:
        device = tgt.device
        tgt_mask = self.cell.generate_square_subsequent_mask(tgt.size(1)).to(device)

        tgt = self.target_emb(tgt)
        output = self.decoder(
            tgt=tgt,
            tgt_mask=tgt_mask,
            memory=memory,
        )
        output = self.linear(output)
        return output


def train(model: Transformer, dataloader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler, device: device, steps: int, ls_eps: float):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_id, label_smoothing=ls_eps)

    for step in range(steps):
        total_loss = 0
        for src, tgt_input, tgt in dataloader:
            src, tgt_input, tgt = src.to(device), tgt_input.to(device), tgt.to(device)
            pred = model(src, tgt_input)
            loss = criterion(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f'Step: {step + 1}, Loss: {total_loss:.6f}')


def predict(model: Transformer, dataloader: DataLoader, device: device) -> list[np.ndarray]:
    preds = []
    model.eval()

    with torch.no_grad():
        for src, tgt in dataloader:
            memory = model.encode(src.to(device))
            pred = beam_search(model, memory, MAX_LEN)


def beam_search(model, memory, max_len, beam_size = 4, alpha = 0.6):
    beams = [(torch.tensor([SOS_id], device=device), 0.0)]

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1].item() == EOS_id:
                new_beams.append((seq, score))
                continue

            logits = model.decode_step(memory, seq.unsqueeze(0))  # [1, T, tgt_vocab_size]
            log_probs = F.log_softmax(logits[0, -1], dim=-1)

            topk_log_probs, topk_ids = log_probs.topk(beam_size)

            for log_prob, token_id in zip(topk_log_probs, topk_ids):
                new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                new_score = score + log_prob.item()
                new_beams.append((new_seq, new_score))

        beams = sorted(
            new_beams,
            key=lambda x: x[1] / ((len(x[0]) + 5) / 6) ** alpha,
            reverse=True
        )[:beam_size]

        if all(seq[-1].item() == EOS_id for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return best_seq
