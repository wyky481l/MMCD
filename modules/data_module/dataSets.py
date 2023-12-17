import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from modules.sequence.encode import index_to_onehot
from util.embed.sequence import fasta_to_index


class MMCDDataset(Dataset):
    def __init__(self, AMP_id_list, AMP_fasta_list, AMP_strut_data, nonAMP_id_list, nonAMP_fasta_list,
                 nonAMP_strut_data):
        self.AMP_id_list = AMP_id_list
        self.AMP_fasta_list = AMP_fasta_list
        self.AMP_strut_data = AMP_strut_data

        self.nonAMP_id_list = nonAMP_id_list
        self.nonAMP_fasta_list = nonAMP_fasta_list
        self.nonAMP_strut_data = nonAMP_strut_data

    def __getitem__(self, index):
        AMP_id = self.AMP_id_list[index]
        AMP_fasta = self.AMP_fasta_list[index]
        AMP_pos = torch.tensor(self.AMP_strut_data[AMP_id])

        nonAMP_id = self.nonAMP_id_list[index]
        nonAMP_fasta = self.nonAMP_fasta_list[index]
        nonAMP_pos = torch.tensor(self.nonAMP_strut_data[nonAMP_id])

        AMP_logit = index_to_onehot(fasta_to_index(AMP_fasta))
        nonAMP_logit = index_to_onehot(fasta_to_index(nonAMP_fasta))

        data = Data(x=index, pos=AMP_pos, fasta=AMP_fasta, logit=AMP_logit, nonamp_pos=nonAMP_pos,
                    nonamp_fasta=nonAMP_fasta, nonamp_logit=nonAMP_logit)
        return data

    def __len__(self):
        return len(self.AMP_fasta_list)
