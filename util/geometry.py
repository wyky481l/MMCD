import Bio
import torch.nn.functional as F
from PeptideBuilder import make_res_of_type
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from PeptideBuilder import Geometry
import PeptideBuilder
from util.embed.structure import *


class Peptide(object):
    """
        # Backbone description
        CA_C_N_angle
        C_N_CA_angle
        N_CA_C_angle
        CA_C_O_angle
        N_CA_C_O_diangle
        phi
        psi
        omega
        CA_N_length
        CA_C_length
        C_O_length

        # atom position list
        pos_CA
        pos_C
        pos_N
        pos_O

        # Backbone coord
        CA_coord
        C_coord
        N_coord
        O_coord
    """

    def __init__(self, pos_list, fasta):
        self.pos_CA = pos_list[:, 0, :]
        self.pos_C = pos_list[:, 1, :]
        self.pos_N = pos_list[:, 2, :]
        self.pos_O = pos_list[:, 3, :]
        self.fasta = fasta
        self.structure = None

        self.CA_C_N_angle = F.pad(
            calc_angle(self.pos_CA[:-1, :], self.pos_C[:-1, :], self.pos_N[1:, :]),
            pad=(0, 1), value=-1,
        )

        self.C_N_CA_angle = F.pad(
            calc_angle(self.pos_C[:-1, :], self.pos_N[1:, :], self.pos_CA[1:, :]),
            pad=(0, 1), value=-1,
        )

        self.N_CA_C_angle = calc_angle(self.pos_N[:, :], self.pos_CA[:, :], self.pos_C[:, :])
        self.CA_C_O_angle = calc_angle(self.pos_CA[:, :], self.pos_C[:, :], self.pos_O[:, :])
        self.N_CA_C_O_diangle = calc_dihedral(self.pos_N[:, :], self.pos_CA[:, :], self.pos_C[:, :], self.pos_O[:, :])

        self.phi = F.pad(
            calc_dihedral(self.pos_C[:-1, :], self.pos_N[1:, :], self.pos_CA[1:, :], self.pos_C[1:, :]),
            pad=(1, 0), value=0,
        )

        self.psi = F.pad(
            calc_dihedral(self.pos_N[:-1, :], self.pos_CA[:-1, :], self.pos_C[:-1, :], self.pos_N[1:, :]),
            pad=(0, 1), value=0,
        )

        self.omega = F.pad(
            calc_dihedral(self.pos_CA[:-1, :], self.pos_C[:-1, :], self.pos_N[1:, :], self.pos_CA[1:, :]),
            pad=(0, 1), value=-1,
        )

        self.CA_N_length = calc_distance(self.pos_CA[:, :], self.pos_N[:, :])
        self.CA_C_length = calc_distance(self.pos_CA[:, :], self.pos_C[:, :])
        self.C_O_length = calc_distance(self.pos_C[:, :], self.pos_O[:, :])

    def reconstruct(self):
        for index, aa in enumerate(self.fasta):
            residue = Geometry.geometry(aa)
            residue = self.update(residue, index)

            if index == 0:
                structure = self.initialize(residue)
            else:
                structure = self.add_reside(structure, residue, index)

        structure = PeptideBuilder.add_terminal_OXT(structure)

        self.structure = structure

        return structure

    def initialize(self, residue, index=0):

        segID = 1
        N_coord = self.pos_N[index, :]
        CA_coord = self.pos_CA[index, :]
        C_coord = self.pos_C[index, :]
        O_coord = self.pos_O[index, :]

        N = Atom("N", N_coord, 0.0, 1.0, " ", " N", 0, "N")
        CA = Atom("CA", CA_coord, 0.0, 1.0, " ", " CA", 0, "C")
        C = Atom("C", C_coord, 0.0, 1.0, " ", " C", 0, "C")
        O = Atom("O", O_coord, 0.0, 1.0, " ", " O", 0, "O")

        res = make_res_of_type(segID, N, CA, C, O, residue)

        cha = Chain("A")
        cha.add(res)

        mod = Model(0)
        mod.add(cha)

        struct = Structure("X")
        struct.add(mod)
        return struct

    def update(self, residue, index):
        if self.CA_C_N_angle[index] != -1:
            residue.CA_C_N_angle = self.CA_C_N_angle[index]

        if self.C_N_CA_angle[index] != -1:
            residue.C_N_CA_angle = self.C_N_CA_angle[index]

        residue.N_CA_C_angle = self.N_CA_C_angle[index]
        residue.CA_C_O_angle = self.CA_C_O_angle[index]
        residue.N_CA_C_O_diangle = self.N_CA_C_O_diangle[index]

        if self.phi[index] != -1:
            residue.phi = self.phi[index]

        if self.psi[index] != -1:
            residue.psi = self.psi[index]

        if self.omega[index] != -1:
            residue.omega = self.omega[index]

        residue.CA_N_length = self.CA_N_length[index]
        residue.CA_C_length = self.CA_C_length[index]
        residue.C_O_length = self.C_O_length[index]

        return residue

    def add_reside(self, structure, residue, index):
        segID = index + 1
        N_coord = self.pos_N[index, :]
        CA_coord = self.pos_CA[index, :]
        C_coord = self.pos_C[index, :]
        O_coord = self.pos_O[index, :]

        N = Atom("N", N_coord, 0.0, 1.0, " ", " N", 0, "N")
        CA = Atom("CA", CA_coord, 0.0, 1.0, " ", " CA", 0, "C")
        C = Atom("C", C_coord, 0.0, 1.0, " ", " C", 0, "C")
        O = Atom("O", O_coord, 0.0, 1.0, " ", " O", 0, "O")

        res = make_res_of_type(segID, N, CA, C, O, residue)
        structure[0]["A"].add(res)
        return structure

    def output_to_pdb(self, pdb_name):
        pdb_dir = "data/output/pdb"
        out = Bio.PDB.PDBIO()
        out.set_structure(self.structure)
        pdb_path = "{}/{}.pdb".format(pdb_dir, pdb_name)
        out.save(pdb_path)
        print("generate " + pdb_path)
