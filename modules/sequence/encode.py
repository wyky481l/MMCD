import math
import time
import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from torch import nn
import torch.nn.functional as F
import torch
from util.constant import aa_count_freq
from util.embed.embedding import sequence_embedding
from pytorch_metric_learning import miners, distances, losses, reducers
from Bio.Seq import Seq
from Bio import SeqIO


class SelfAttention(nn.Module):
    def __init__(self,
                 n_emb,
                 n_head,
                 attn_drop=0.1,
                 resid_drop=0.1
                 ):
        super().__init__()

        # key, query, value projections for all heads
        self.key = nn.Linear(n_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)

        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        # output projection
        self.proj = nn.Linear(n_emb, n_emb)
        self.n_head = n_head

    def forward(self, x, batch, condition=None):
        token_list = []
        attn_list = []

        seq_length_list = torch.bincount(batch)

        for i in range(len(seq_length_list)):
            x_i = x[batch == i,]
            T, C = x_i.size()

            k = self.key(x_i).view(T, self.n_head, C // self.n_head).transpose(0, 1)
            q = self.query(x_i).view(T, self.n_head, C // self.n_head).transpose(0, 1)
            v = self.value(x_i).view(T, self.n_head, C // self.n_head).transpose(0, 1)
            # [n_head, T, C]
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            attention = F.softmax(attention, dim=-1)

            attention = self.attn_drop(attention)
            y_i = attention @ v
            # [n_head, T, T] * [n_head, T, C] -> [n_head, T, C]
            y_i = y_i.transpose(0, 1).contiguous().view(T, C)

            attention = attention.mean(dim=0, keepdim=False)

            # output projection
            y_i = self.resid_drop(self.proj(y_i))

            token_list.append(y_i)
            attn_list.append(attention)

        y = torch.concat(token_list, dim=0)

        return y, attn_list


class CrossAttention(nn.Module):
    def __init__(self,
                 n_emb,
                 n_cond_emb,
                 n_head,
                 attn_drop=0.1,
                 resid_drop=0.1,
                 ):
        super().__init__()

        # key, query, value projections for all heads
        self.key = nn.Linear(n_cond_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_cond_emb, n_emb)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        # output projection
        self.proj = nn.Linear(n_emb, n_emb)

        self.n_head = n_head

    def forward(self, x, condition, batch=None):
        T, C = x.size()

        T_cond, _ = condition.size()
        k = self.key(condition).view(T_cond, self.n_head, C // self.n_head).transpose(0, 1)
        q = self.query(x).view(T, self.n_head, C // self.n_head).transpose(0, 1)
        v = self.value(condition).view(T_cond, self.n_head, C // self.n_head).transpose(0, 1)

        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(T, C)
        attention = attention.mean(dim=0, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, attention


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=2000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class SeqFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="silu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, input_dim * 4, input_dim * 2, input_dim, input_dim // 2, input_dim // 4, output_dim]

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 1):
            self.layers.append(nn.Linear(self.dim_list[i], self.dim_list[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class MetricLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        self.miner = miners.MultiSimilarityMiner()
        self.distances = distances.CosineSimilarity()
        self.reducer = reducers.MeanReducer()

        self.metric_trip_loss = losses.TripletMarginLoss(
            distance=self.distances,
            reducer=self.reducer,
        )

        self.metric_cont_loss = losses.ContrastiveLoss(
            distance=self.distances,
            pos_margin=1,
            neg_margin=0
        )

        self.cont_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, AMP_emb, nonAMP_emb, loss_type="cont"):
        assert loss_type in {"cont", "metric_cont", "metric_trip"}, print("metric_loss_type error")

        emb = torch.concat((AMP_emb, nonAMP_emb), dim=0)

        AMP_label = torch.ones(len(AMP_emb), device=AMP_emb.device)
        nonAMP_label = torch.zeros(len(nonAMP_emb), device=nonAMP_emb.device)
        label = torch.concat((AMP_label, nonAMP_label))

        if loss_type == "cont":
            loss = self.cont_loss(emb, label)
        elif loss_type == "metric_cont":
            loss = self.metric_cont_loss(emb, label)
        else:
            hard_pairs = self.miner(emb, label)
            loss = self.trip_loss(emb, label, hard_pairs)

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        n = labels.shape[0]
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        mask_pos = torch.ones_like(similarity_matrix, device=features.device) * (
            labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask_neg = torch.ones_like(mask_pos, device=features.device) - mask_pos

        similarity_matrix = torch.exp(similarity_matrix / self.T)

        mask_diag = (torch.ones(n, n) - torch.eye(n, n)).to(features.device)
        similarity_matrix = similarity_matrix * mask_diag

        sim_pos = mask_pos * similarity_matrix
        sim_neg = similarity_matrix - sim_pos
        sim_neg = torch.sum(sim_neg, dim=1).repeat(n, 1).T
        sim_total = sim_pos + sim_neg

        loss = torch.div(sim_pos, sim_total)
        loss = mask_neg + loss + torch.eye(n, n, device=features.device)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)

        return loss


class MatchLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, feature_left, feature_right, match_type="graph"):
        assert match_type in {"node", "graph"}, print("match_type error")
        device = feature_left.device

        if match_type == "node":
            similarity = F.cosine_similarity(feature_left, feature_right, dim=1).to(device)
            similarity = torch.exp(similarity / self.T)
            loss = torch.mean(-torch.log(similarity))
        else:
            n = len(feature_left)
            similarity = F.cosine_similarity(feature_left.unsqueeze(1), feature_right.unsqueeze(0), dim=2).to(device)
            similarity = torch.exp(similarity / self.T)

            mask_pos = torch.eye(n, n, device=device, dtype=bool)
            sim_pos = torch.masked_select(similarity, mask_pos)

            sim_total_row = torch.sum(similarity, dim=0)
            loss_row = torch.div(sim_pos, sim_total_row)
            loss_row = -torch.log(loss_row)

            sim_total_col = torch.sum(similarity, dim=1)
            loss_col = torch.div(sim_pos, sim_total_col)
            loss_col = -torch.log(loss_col)

            loss = loss_row + loss_col
            loss = torch.sum(loss) / (2 * n)

        return loss


class MetricPredictorLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.activate = nn.ReLU()

        self.project_emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            self.activate,
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            self.activate,
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, feature):
        out = self.project_emb(feature)
        return out


def index_to_onehot(x, num_classes=20):
    x = torch.tensor(x)

    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'

    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)

    return x_onehot.float()


def logit_to_index(logit_p, random_state=False):
    if random_state:
        D = torch.distributions.Categorical(logit_p)
        token_index = D.sample()
    else:
        token_index = logit_p.argmax(dim=-1)

    return token_index


# sum P(x)log(P(x)/Q(x))
def multinomial_kl(prob1, prob2):
    # kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    prob1 = prob1.softmax(dim=-1)
    prob2 = prob2.softmax(dim=-1)

    kl = (prob1 * torch.log(prob1 / prob2)).sum(dim=-1)

    return kl.mean()


def get_time_steps(n_seq, n_timestep, device=None):
    # time_step = torch.randint(
    #     0, n_timestep, size=(n_seq // 2 + 1,), device=device)
    # time_step = torch.cat(
    #     [time_step, n_timestep - time_step - 1], dim=0)[:n_seq]

    time_step = torch.randint(1, n_timestep, size=(n_seq,), device=device)

    return time_step


def get_seq_noise(seq_len=1, device=None, noise_state="dmd", n_class=20):
    # dud: discrete uniform distribution
    # dmd: discrete marginal distribution

    if noise_state == "dud":
        noise = torch.ones([seq_len, n_class], device=device) / n_class
    else:
        noise = torch.tensor(aa_count_freq, device=device).unsqueeze(dim=0).repeat(seq_len, 1)

    return noise


# Qt = alphas_bar * I + (1 - alphas_bar) * K
def get_Qt_weight(alphas_bar, noise, batch, device, n_class=20):
    # Q_weight = [bar_t * torch.eye(self.n_class, device=self.device) + (1 - bar_t) * noise for bar_t in
    #            token_alphas_bar]
    # Q_weight = torch.stack(Q_weight).float()
    Qt_weight = [bar_t * torch.eye(n_class, device=device) + (1 - bar_t) * noise for bar_t in
                 alphas_bar]
    Qt_weight = torch.stack(Qt_weight).float()
    Qt_weight = Qt_weight.index_select(0, batch)
    # [N,20,20]
    return Qt_weight


def batch_sequence_embedding(seq_logit, batch, batch_size, device):
    seq_emd_list = []

    for i in range(batch_size):
        seq_index = logit_to_index(seq_logit[batch == i,])
        seq_emd = sequence_embedding(index=seq_index)
        seq_emd_list.append(seq_emd)

    seq_emd_list = np.concatenate(seq_emd_list, axis=0)
    out = torch.tensor(seq_emd_list, device=device).float()

    return out


def token_aa_acc(pred, real, device):
    y_pred = torch.argmax(pred, dim=-1)
    y_real = torch.argmax(real, dim=-1)
    score = torch.sum(torch.tensor(y_pred == y_real, device=device)) / len(y_pred)

    return score


def get_seq_batch_info(data, device):
    nonAMP_fasta_list = data.nonamp_seq
    y_batch = []

    for i in range(len(nonAMP_fasta_list)):
        seq_length = len(nonAMP_fasta_list[i])
        seq_value = torch.full([seq_length], i, device=device)
        y_batch.append(seq_value)

    y_batch = torch.concat(y_batch)
    data.y_batch = y_batch

    return data


def get_batch_info(data_list, device):
    batch = []

    for i in range(len(data_list)):
        data_length = len(data_list[i])
        data_value = torch.full([data_length], i, device=device)
        batch.append(data_value)

    batch = torch.concat(batch)

    return batch


def get_struct_batch_info(fasta_list, device):
    batch = []

    for i in range(len(fasta_list)):
        fasta_length = len(fasta_list[i])
        fasta_value = torch.full([fasta_length], i, device=device)
        batch.append(fasta_value)

    batch = torch.concat(batch)

    return batch


def get_attn_emb(seq_emb, seq_attn, batch):
    seq_size = len(seq_attn)
    seq_attn_emb_list = []

    for index in range(seq_size):
        attn = seq_attn[index].mean(dim=0)
        emb = seq_emb[batch == index,]
        attn_emb = attn @ emb

        attn_emb = attn_emb.unsqueeze(0)
        seq_attn_emb_list.append(attn_emb)

    return seq_attn_emb_list


def save_output_seq(out_seq_list):
    record_list = []

    for i, seq_str in enumerate(out_seq_list):
        seq_id = "AMP_{}".format(i)
        seq_desc = ""

        record = SeqRecord(Seq(seq_str), id=seq_id, description=seq_desc)
        record_list.append(record)

    time_str = str(pd.Timestamp.now())[:16]
    record_path = "data/output/fasta/AMP_{}.fasta".format(time_str)
    print("generate " + record_path)
    SeqIO.write(record_list, record_path, "fasta")

    return record_path
