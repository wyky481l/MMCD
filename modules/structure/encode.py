import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from util.embed.embedding import structure_embedding
from util.embed.structure import calc_distance_residue, get_aaindex_embedding, get_node_pairs, get_contact_embedding


class TimeEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, embed_size, 2).float() / embed_size)
        )

    def forward(self, time_step, batch, edge_index=None):
        self.inv_freq = self.inv_freq.to(time_step.device)
        if batch is not None:

            if edge_index is not None:
                edge_graph = batch.index_select(0, edge_index[0])
                edge_time_step = time_step.index_select(0, edge_graph)
                time_step = edge_time_step.unsqueeze(dim=-1)
            else:
                node_time_step = time_step.index_select(0, batch)
                time_step = node_time_step.unsqueeze(dim=-1)
        else:
            time_step = time_step.unsqueeze(dim=-1)

        pos_enc_a = torch.sin(time_step.repeat(1, self.embed_size // 2) * self.inv_freq)
        pos_enc_b = torch.cos(time_step.repeat(1, self.embed_size // 2) * self.inv_freq)
        pos_enc = torch.cat((pos_enc_a, pos_enc_b), dim=-1)

        return pos_enc


class MLPEdgeEncoder(nn.Module):
    def __init__(self, edge_dim=8, output_dim=32):
        super().__init__()
        self.input_dim = edge_dim + 1
        self.edge_emb = nn.Sequential(
            nn.Linear(self.input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, edge_attr, edge_length):
        edge_attr_feat = torch.cat([edge_attr, edge_length], dim=-1)
        out = self.edge_emb(edge_attr_feat)
        return out


class MLP_block(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation="relu", dropout=0.1, num_block=32):
        super(MLP_block, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim * 2)
        self.time_embed = TimeEncoder(hidden_dim * 2)

        self.dims = [hidden_dim * 2 for i in range(num_block)]
        self.dims += [hidden_dim, hidden_dim // 2, hidden_dim // 4, 4 * 3]

        self.lns = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(self.dims) - 1):
            self.lns.append(nn.LayerNorm(self.dims[i]))
            self.mlps.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, time_step, batch):
        x = self.input(x)
        time_embed = self.time_embed(time_step, batch)
        x = x + time_embed

        for i, (ln, mlp) in enumerate(zip(self.lns, self.mlps)):
            x = mlp(ln(x))

            if i < len(self.mlps) - 1:
                if self.activation:
                    x_mlp = self.activation(x_mlp)
                if self.dropout:
                    x_mlp = self.dropout(x_mlp)

            if i < len(self.mlps) - 4:
                x = x + x_mlp
            else:
                x = x_mlp

        return x


class StructFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation="silu", dropout=0.1):
        super(StructFFN, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim * 2)
        self.time_embed = TimeEncoder(hidden_dim * 2)

        self.dims = [hidden_dim * 2, hidden_dim, hidden_dim // 2, hidden_dim // 4, 4 * 3]

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.lns = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(self.dims) - 1):
            self.lns.append(nn.LayerNorm(self.dims[i]))
            self.mlps.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, x, time_step, batch):
        x = self.input(x)
        time_embed = self.time_embed(time_step, batch)
        x = x + time_embed

        for i, (ln, mlp) in enumerate(zip(self.lns, self.mlps)):
            x = mlp(ln(x))

            if i < len(self.mlps) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)

        return x


def atom_pair_feature(node_attr, edge_index, edge_attr, pos):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    edge_length = calc_distance_residue(pos, edge_index)
    atom_embedding = get_backbone_atom_embedding()

    edge_attr = torch.cat([h_row * h_col, edge_attr], dim=-1)
    # [N,128+32]
    edge_attr = torch.unsqueeze(edge_attr, dim=1).repeat(1, 4, 1)
    # [N,160] -> [N,4,160]

    edge_length_ = torch.unsqueeze(edge_length, dim=-1)
    # [N,4] -> [N,4,1]
    atom_embedding = torch.unsqueeze(atom_embedding, dim=0).repeat(len(edge_length), 1, 1).to(pos.device)
    # [4,4] -> [N,4,4]
    atom_embedding = torch.cat([edge_length_, atom_embedding], dim=-1)
    # [N,4,4] -> [N,4,5]

    atom_edge_attr = torch.cat([edge_attr, atom_embedding], dim=-1).to(pos.device)
    # [N,4,165]

    return atom_edge_attr, edge_length


def get_backbone_atom_embedding(atom_size=4):
    backbone_atom = F.one_hot(torch.arange(atom_size))
    return backbone_atom


def get_batch_structure_embedding(pos, batch, fasta_list, device, constant_data, threshold=5):
    total_fasta = "".join(fasta_list)
    node_emb, edge_index, edge_emb, edge_length = structure_embedding(pos, fasta=total_fasta, device=device,
                                                                      batch=batch, threshold=threshold,
                                                                      constant_data=constant_data)

    return node_emb, edge_index, edge_emb, edge_length
