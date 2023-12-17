import math
from tqdm import tqdm

from util.constant import *
import numpy as np
from Bio.PDB.PDBParser import PDBParser
import torch

from util.embed.sequence import fasta_to_index


def get_aaindex_embedding(fasta, device=None):
    aa_index = torch.tensor(np.array(aaindex.values[:, 1:], dtype=float), device=device)
    aa_helix = torch.tensor(AA_helix, device=device).unsqueeze(dim=0)
    aa_info = torch.cat((aa_index, aa_helix), dim=0)

    fasta_index = torch.tensor(fasta_to_index(fasta), device=device)
    embedding = torch.index_select(aa_info, dim=1, index=fasta_index).T

    return embedding


def get_contact_embedding2(fasta, node_pairs, constant_data, edge_length):
    device = node_pairs.device
    fasta_index = torch.tensor(fasta_to_index(fasta), device=device)

    left_aa = torch.index_select(fasta_index, dim=0, index=node_pairs[:, 0])
    right_aa = torch.index_select(fasta_index, dim=0, index=node_pairs[:, 1])

    dpsp_score_list = []

    for i, pair in enumerate(node_pairs):
        left_aa_i = left_aa[i]
        right_aa_i = right_aa[i]
        distance = edge_length[i]

        aa_type = AA_type[left_aa_i]
        aa_index = right_aa_i

        if distance <= 5:
            dpsp_score = dpsp_0_5[aa_type][aa_index]
        elif distance <= 7.5:
            dpsp_score = dpsp_5_7[aa_type][aa_index]
        elif distance <= 10:
            dpsp_score = dpsp_7_10[aa_type][aa_index]
        elif distance <= 12:
            dpsp_score = dpsp_10_12[aa_type][aa_index]
        else:
            dpsp_score = dpsp_over_12[aa_type][aa_index]

        dpsp_score_list.append(dpsp_score)

    constant_data = constant_data.to(device)

    score = constant_data[:, left_aa, right_aa].T

    dpsp = torch.tensor(np.stack(dpsp_score_list), device=device).unsqueeze(dim=-1).float()
    score = torch.concat([dpsp, score], dim=-1)

    return score


def get_contact_embedding(fasta, node_pairs, constant_data, edge_length):
    device = node_pairs.device
    fasta_index = torch.tensor(fasta_to_index(fasta), device=device)

    left_aa = torch.index_select(fasta_index, dim=0, index=node_pairs[:, 0])
    right_aa = torch.index_select(fasta_index, dim=0, index=node_pairs[:, 1])

    constant_data = constant_data.to(device)
    score = constant_data[:, left_aa, right_aa].T

    return score


def get_position_from_pdb(pdb_path):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure("PDB_ID", pdb_path)
    residue_list = structure.get_residues()

    pos_list = []
    for residue in residue_list:
        pos_list.append([[residue['CA'].coord], [residue['C'].coord], [residue['N'].coord], [residue['O'].coord]])

    pos_list = np.array(pos_list).reshape((-1, 4, 3))

    return pos_list


def get_node_pairs(coords, threshold):
    node_pairs = []
    dist_list = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            dist = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2 + \
                             (coords[i][2] - coords[j][2]) ** 2)
            if 0 < dist < threshold:
                node_pairs.append([i, j])
                dist_list.append(dist)

    return node_pairs, dist_list


def calc_distance(pos_1, pos_2):
    dist = torch.sqrt(torch.sum((pos_1 - pos_2) ** 2, dim=-1))
    return dist


def calc_distance_residue(pos, edge_index):
    return torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=-1)


def calc_angle(pos_0, pos_1, pos_2):
    v1 = pos_0 - pos_1
    v2 = pos_2 - pos_1

    n1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
    n2 = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True)

    angle = torch.acos((n1 * n2).sum(-1))
    angle = torch.rad2deg(angle)

    return angle


def calc_dihedral(pos_0, pos_1, pos_2, pos_3):
    v0 = pos_2 - pos_1
    v1 = pos_0 - pos_1
    v2 = pos_3 - pos_2

    u1 = torch.linalg.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.linalg.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)

    dihedral = torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    dihedral = torch.nan_to_num(dihedral)
    dihedral = torch.rad2deg(dihedral)
    # 将角度数值抓换为弧度
    return dihedral
