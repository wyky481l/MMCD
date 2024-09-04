import math
from torch import nn
import torch.nn.functional as F
import torch
from modules.sequence.encode import SelfAttention, SinusoidalPosEmb


# Transformer block
class SeqBlock(nn.Module):
    def __init__(self,
                 n_emb,
                 n_head,
                 attn_drop,
                 resid_drop,
                 n_diff_step,
                 n_seq_max,
                 emb_type
                 ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_emb, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(n_emb)
        self.dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()             

        self.attn = SelfAttention(
            n_emb=n_emb,
            n_head=n_head,
            attn_drop=attn_drop,
            resid_drop=attn_drop
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, 2 * n_emb),
            nn.GELU(),
            nn.Linear(2 * n_emb, n_emb),
            nn.Dropout(resid_drop),
        )

        if emb_type == "pos_emb":
            self.emb_t = SinusoidalPosEmb(n_diff_step, n_emb)
            self.emb_pos = SinusoidalPosEmb(n_seq_max, n_emb)
        else:
            self.emb_t = nn.Embedding(n_diff_step, n_emb)
            self.emb_pos = nn.Embedding(n_seq_max, n_emb)

        self.silu = nn.SiLU()
        self.linear_t = nn.Linear(n_emb, n_emb)
        self.linear_pos = nn.Linear(n_emb, n_emb)

    def forward(self, x, time_step, batch):
        time_emb = self.silu(self.linear_t(self.emb_t(time_step)))

        seq_length_list = torch.bincount(batch)

        pos_emb = [torch.range(0, seq_length - 1, device=x.device) for seq_length in seq_length_list]
        pos_emb = torch.concat(pos_emb, dim=-1)
        pos_emb = self.silu(self.linear_pos(self.emb_pos(pos_emb)))

        x = x + time_emb + pos_emb

        a, att = self.attn(x, batch)
        x = self.dropout(x + a)
        x = self.ln1(x)

        x = self.dropout(x + self.ffn(x))
        x = self.ln2(x)

        return x, att


class SeqTransformer(nn.Module):
    def __init__(
            self,
            input_dim=None,
            output_dim=128,
            n_emb=128,
            n_head=16,
            attn_drop=0.1,
            resid_drop=0.1,
            n_diff_step=500,
            n_block=8,
            emb_type="pos_emb",
            n_seq_max=50

    ):
        super().__init__()

        self.cont_emb = nn.Linear(input_dim, n_emb)
        self.n_block = n_block

        self.output_emb = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, output_dim),
        )

        self.blocks = nn.Sequential(*[SeqBlock(
            n_emb=n_emb,
            n_head=n_head,
            attn_drop=attn_drop,
            resid_drop=resid_drop,
            n_diff_step=n_diff_step,
            emb_type=emb_type,
            n_seq_max=n_seq_max
        ) for n in range(n_block)])

    def forward(self, x, time_step, batch=None):
        x_emb = self.cont_emb(x)

        for index in range(self.n_block):
            x_emb, attn_weight = self.blocks[index](x_emb, time_step, batch)

        output = self.output_emb(x_emb)
        return output
