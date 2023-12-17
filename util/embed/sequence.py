import numpy as np
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from util.constant import *


def fasta_to_index(fasta):
    index_list = [AA_dict[aa] for aa in fasta]
    return index_list


def index_to_fasta(index_list):
    fasta = [AA_type[index] for index in index_list]
    fasta = "".join(fasta)
    return fasta


def onehot_encoding(seq):
    encoding_map = np.eye(len(AA_type))

    residues_map = {}
    for i, aa in enumerate(AA_type):
        residues_map[aa] = encoding_map[i]

    tmp_seq = [residues_map[aa] for aa in seq]
    return np.array(tmp_seq)


def position_encoding(seq_length):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 50 for the short length of peptides.
    """
    d = 20
    b = 50
    N = seq_length
    value = []
    for pos in range(N):
        tmp = []
        for i in range(d // 2):
            tmp.append(pos / (b ** (2 * i / d)))
        value.append(tmp)

    value = np.array(value)
    pos_encoding = np.zeros((N, d))
    pos_encoding[:, 0::2] = np.sin(value[:, :])
    pos_encoding[:, 1::2] = np.cos(value[:, :])

    return pos_encoding


def save_index_to_fasta(index_list, output_path="data/output"):
    record_list = []
    for i, index in enumerate(index_list):
        record_id = "output_fasta_{}".format(i)
        record_seq = index_to_fasta(index)
        record = SeqRecord(Seq(record_seq), id=record_id, description="")
        record_list.append(record)

    output_file = "{}/output.fasta".format(output_path)
    SeqIO.write(record_list, output_file, "fasta")


def load_fasta_to_index(fasta_path):
    index_list = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        # record_id = record.id
        record_seq = record.seq
        index = fasta_to_index(record_seq)
        index_list.append(index)

    return index_list


def get_blosum_embedding_from_peptide(fasta):
    embedding = []
    for aa in fasta:
        embedding = embedding + blosum62[aa]

    return embedding


def get_pam_embedding_from_peptide(fasta):
    embedding = []
    for aa in fasta:
        embedding = embedding + PAM120[aa]

    return embedding


def get_hydrophobicity_embedding_from_peptide(fasta, scale=100):
    embedding = []
    for aa in fasta:
        embedding = embedding + [score / scale for score in hydrophobicity[aa]]

    return embedding


def load_pssm_embedding(pssmDir, pssm_name):
    pssm_path = "{}/{}.pssm".format(pssmDir, pssm_name)
    assert os.path.exists(pssm_path), 'pssm file {} does not exist'.format(pssm_name)

    with open(pssm_path) as f:
        records = f.readlines()[3: -6]

    pssmMatrix = []
    for line in records:
        array = line.strip().split()
        pssmMatrix.append([int(num) for num in array[2:22]])

    return pssmMatrix


def get_pssm_embedding_from_peptide(fasta, db_type="nrdb90", tmp_dir="data/tmp"):
    record_id = "tmp_id"
    record_seq = fasta
    record_name = "tmp_seq"

    assert tmp_dir is not None, "please set the tmp path"
    assert db_type is not None, "please select the psiblast database"

    if db_type == "nrdb90":
        db_path = "data/psiblast/nrdb90/nrdb90"
    else:
        db_path = "data/psiblast/nr/nr"

    save_fasta = SeqRecord(Seq(record_seq), id=record_id, description="")
    SeqIO.write(save_fasta, "{}/{}.fasta".format(tmp_dir, record_name), "fasta")

    input_path = "{}/{}.fasta".format(tmp_dir, record_name)
    output_path = "{}/{}.pssm".format(tmp_dir, record_name)
    assert os.path.exists(output_path), 'pssm file output error'

    log_path = "{}/{}.xml".format(tmp_dir, record_name)
    command = "psiblast -query {} -db {} -evalue 0.001 -num_iterations 3 -num_threads 6 -out_ascii_pssm {} -out {}".format(
        input_path, db_path, output_path, log_path)
    os.system(command)

    embedding = load_pssm_embedding(tmp_dir, record_name)

    return embedding


def get_embedding_form_peptide(fasta=None, encode_type=None):
    assert encode_type in {"pssm", "blosum", "pam", "hydrophobicity"}, "embedding type error"

    if encode_type == "pssm":
        embedding = get_pssm_embedding_from_peptide(fasta)
    elif encode_type == "blosum":
        embedding = get_blosum_embedding_from_peptide(fasta)
    elif encode_type == "pam":
        embedding = get_pam_embedding_from_peptide(fasta)
    else:
        embedding = get_hydrophobicity_embedding_from_peptide(fasta)

    return embedding


def get_bio_embedding_for_sequence(fasta, encode_type=None):
    if encode_type is None:
        encode_type = ["blosum", "pam", "hydrophobicity"]

    embedding = []
    for aa_type in encode_type:
        embedding = embedding + get_embedding_form_peptide(fasta=fasta, encode_type=aa_type)

    embedding = np.array(embedding).reshape(len(fasta), -1)

    return embedding
