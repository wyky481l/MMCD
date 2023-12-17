import numpy as np
import torch
from torch_geometric.nn import radius_graph
from tqdm import tqdm
from util.embed.sequence import *
from util.embed.structure import *
import pickle
import torch.nn.functional as F


def sequence_embedding_from_fastaFile(pdb_type):
    assert pdb_type in {"AMP", "nonAMP"}, print("error type for input")

    if pdb_type == "AMP":
        fasta_path = "data/source/fasta/AMP.fasta"
        data_pickle = "data/source/AMP_sequence.pickle"
    else:
        fasta_path = "data/source/fasta/nonAMP.fasta"
        data_pickle = "data/source/nonAMP_sequence.pickle"

    if os.path.exists(data_pickle):
        with open(data_pickle, 'rb') as f:
            embedding_list = pickle.load(f)
        print(data_pickle + " exist")
        return embedding_list

    fasta_list = SeqIO.parse(fasta_path, "fasta")
    embedding_list = {}

    for fasta in tqdm(fasta_list):
        fasta_id = fasta.id
        fasta_seq = fasta.seq
        sequence_data = sequence_embedding(fasta_seq)
        embedding_list[fasta_id] = sequence_data

    with open(data_pickle, 'wb') as f:
        print(data_pickle + " save")
        pickle.dump(embedding_list, f)

    return embedding_list


def sequence_embedding(fasta=None, index=None):
    if fasta is None:
        assert index is not None, "sequence_embedding error"
        fasta = index_to_fasta(index)

    embedding_1 = onehot_encoding(fasta)
    embedding_2 = get_bio_embedding_for_sequence(fasta)
    embedding = np.concatenate((embedding_1, embedding_2), axis=1)

    return embedding


def structure_embeddings_from_pdb(pdb_type):
    assert pdb_type in {"AMP", "nonAMP"}, print("error type for input")

    if pdb_type == "AMP":
        pdb_dir = "data/source/pdb/AMP"
        fasta_path = "data/source/fasta/AMP.fasta"
        data_pickle = "data/source/AMP_structure.pickle"
    else:
        pdb_dir = "data/source/pdb/nonAMP"
        fasta_path = "data/source/fasta/nonAMP.fasta"
        data_pickle = "data/source/nonAMP_structure.pickle"

    if os.path.exists(data_pickle):
        with open(data_pickle, 'rb') as f:
            embedding_list = pickle.load(f)
        print(data_pickle + " exist")
        return embedding_list

    fasta_list = SeqIO.parse(fasta_path, "fasta")
    embedding_list = {}
    for fasta in tqdm(fasta_list):
        fasta_id = fasta.id
        # fasta_seq = fasta.seq

        pos_list = get_position_from_pdb("{}/{}.pdb".format(pdb_dir, fasta_id))

        embedding_list[fasta_id] = pos_list

    with open(data_pickle, 'wb') as f:
        pickle.dump(embedding_list, f)

    return embedding_list


def structure_embedding(pos, fasta=None, index=None, threshold=5, device=None, batch=None, constant_data=None):
    # edges are established based on the Euclidean distance being lower than the threshold

    if fasta is None:
        assert index is not None, "structure_embedding error"
        fasta = index_to_fasta(index)

    node_emb = get_aaindex_embedding(fasta, device)

    pos = torch.as_tensor(pos[:, 0, :], device=device)

    edge_index = radius_graph(pos, r=threshold, batch=batch).long()
    edge_length = F.pairwise_distance(pos[edge_index[0]], pos[edge_index[1]], p=2).unsqueeze(dim=-1)

    node_emb = torch.as_tensor(node_emb, device=device).float()
    edge_emb = get_contact_embedding(fasta, edge_index.T, constant_data, edge_length)
    edge_emb = torch.as_tensor(edge_emb, device=device).reshape(len(edge_emb), -1)

    return node_emb, edge_index, edge_emb, edge_length
