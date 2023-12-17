import random
import numpy as np
import pytorch_lightning as pl
from Bio import SeqIO
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from modules.data_module.dataSets import *

from util.embed.embedding import structure_embeddings_from_pdb


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_fasta_list(dataType):
    assert dataType in {"AMP", "nonAMP"}, print("input type should be AMP or nonAMP")

    if dataType == "AMP":
        fasta_path = "data/source/fasta/AMP.fasta"
    else:
        fasta_path = "data/source/fasta/nonAMP.fasta"

    fasta_data = SeqIO.parse(fasta_path, "fasta")
    fasta_id_list = []
    fasta_seq_list = []

    for fasta in tqdm(fasta_data):
        fasta_id = fasta.id
        fasta_seq = str(fasta.seq)

        fasta_id_list.append(fasta_id)
        fasta_seq_list.append(fasta_seq)

    return fasta_id_list, fasta_seq_list


class MMCDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size

        self.AMP_id_list, self.AMP_fasta_list = get_fasta_list("AMP")
        self.nonAMP_id_list, self.nonAMP_fasta_list = get_fasta_list("nonAMP")

        self.AMP_strut_data = structure_embeddings_from_pdb("AMP")
        self.nonAMP_strut_data = structure_embeddings_from_pdb("nonAMP")

        self.dataset = MMCDDataset(self.AMP_id_list,
                                   self.AMP_fasta_list,
                                   self.AMP_strut_data,
                                   self.nonAMP_id_list,
                                   self.nonAMP_fasta_list,
                                   self.nonAMP_strut_data)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
