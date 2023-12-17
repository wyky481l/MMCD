import numpy as np
from Bio.Align import substitution_matrices
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from Bio import SeqIO, Align
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from tqdm import tqdm


def evaluate_fasta(file_path, check_type):
    assert check_type in ["instability", "ez", "TM_tend"], "evaluate type error"
    if check_type == "instability":
        check_funcation = instability_score
    else:
        check_funcation = ez_score

    fasta_list = list(SeqIO.parse(file_path, "fasta"))
    score_list = []
    for fasta in tqdm(fasta_list):
        fasta_seq = str(fasta.seq)
        score = check_funcation(fasta_seq)
        score_list.append(score)

    return np.array(score_list)


# https://doi.org/10.1093/protein/4.2.155
# https://doi.org/10.1093/bioinformatics/btx285 modlamp
def instability_score(fasta):
    desc = GlobalDescriptor(fasta)
    desc.instability_index()
    score = desc.descriptor

    return score.squeeze()


# https://doi.org/10.1016/j.jmb.2006.09.020
# https://academic.oup.com/bioinformatics/article/33/17/2753/3796392
def ez_score(fasta, window=10):
    AMP = PeptideDescriptor(fasta, 'Ez')
    AMP.calculate_global(window)
    score = AMP.descriptor

    return score.squeeze()


def TM_tend_score(fasta, window=7):
    AMP = PeptideDescriptor(fasta, 'TM_tend')
    AMP.calculate_global(window)
    score = AMP.descriptor

    return score.squeeze()


def match_score(fasta):
    AMP_path = "data/source/fasta/AMP.fasta"
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    score_list = []

    for record in SeqIO.parse(AMP_path, "fasta"):
        amp_str = record.seq
        alignments = aligner.align(amp_str, fasta)
        score = alignments.score

        score_list.append(score)

    score_list = np.stack(score_list)
    return score_list.mean()


def match_score_batch(fasta_path):
    fasta_list = list(SeqIO.parse(fasta_path, "fasta"))
    record_list = []

    for index in tqdm(range(len(fasta_list))):
        fasta_id = fasta_list[index].id
        fasta_str = fasta_list[index].seq
        score = match_score(fasta_str)

        record = {"id": fasta_id, "score": score}
        record_list.append(record)

    return record_list
