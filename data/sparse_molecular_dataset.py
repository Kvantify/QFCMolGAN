# https://github.com/nicola-decao/MolGAN/blob/master/utils/sparse_molecular_dataset.py

import pickle
from datetime import datetime
from typing import Any, Callable, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem  # type: ignore

Mol: TypeAlias = Chem.rdchem.Mol


class SparseMolecularDataset:
    smiles: NDArray
    data_S: NDArray
    data_A: NDArray
    data_X: NDArray
    data_D: NDArray
    data_F: NDArray
    data_Le: NDArray
    data_Lv: NDArray

    train_idx: NDArray
    validation_idx: NDArray
    test_idx: NDArray

    def load(self, filename: str, subset: int = 1) -> None:
        with open(filename, "rb") as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(
            self.validation_idx, int(len(self.validation_idx) * subset), replace=False
        )
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    def generate(
        self,
        filename: str,
        add_h: bool = False,
        filters: Callable[[Any], bool] = lambda x: True,
        size: Optional[int] = None,
        validation: float = 0.1,
        test: float = 0.1,
    ) -> None:
        self.log("Extracting {}..".format(filename))

        if filename.endswith(".sdf"):
            data = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
        elif filename.endswith(".smi"):
            data = [Chem.MolFromSmiles(line) for line in open(filename, "r").readlines()]

        data = list(map(Chem.AddHs, data)) if add_h else data
        data = list(filter(filters, data))
        data = data[:size]

        self.log(
            "Extracted {} out of {} molecules {}adding Hydrogen!".format(
                len(data),
                len(Chem.SDMolSupplier(filename)),
                "" if add_h else "not ",
            )
        )

        self._generate_encoders_decoders(data)
        self._generate_AX(data)

        self.vertexes = self.data_F.shape[-2]
        self.features = self.data_F.shape[-1]

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self, input_data: list[Mol]) -> None:
        self.log("Creating atoms encoder and decoder..")
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in input_data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        self.log(
            "Created atoms encoder and decoder with {} atom types and 1 PAD symbol!".format(self.atom_num_types - 1)
        )

        self.log("Creating bonds encoder and decoder..")
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(
            sorted(set(bond.GetBondType() for mol in input_data for bond in mol.GetBonds()))
        )

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log(
            "Created bonds encoder and decoder with {} bond types and 1 PAD symbol!".format(self.bond_num_types - 1)
        )

        self.log("Creating SMILES encoder and decoder..")
        smiles_labels = ["E"] + list(set(c for mol in input_data for c in Chem.MolToSmiles(mol)))
        self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        self.smiles_num_types = len(smiles_labels)
        self.log("Created SMILES encoder and decoder with {} types and 1 PAD symbol!".format(self.smiles_num_types - 1))

    def _generate_AX(self, input_data: list[Mol]) -> None:
        self.log("Creating features and adjacency matrices..")

        data = []
        smiles = []
        data_S = []
        data_A = []
        data_X = []
        data_D = []
        data_F = []
        data_Le = []
        data_Lv = []

        max_length = max(mol.GetNumAtoms() for mol in input_data)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in input_data)

        for _, mol in enumerate(input_data):
            A = self._genA(mol, connected=True, max_length=max_length)
            if A is not None:
                D = np.count_nonzero(A, -1)
                data.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))

                L = D - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)

        self.log(date=False)
        self.log("Created {} features and adjacency matrices  out of {} molecules!".format(len(data), len(input_data)))

        self.data = np.array(data)
        self.smiles = np.array(smiles)
        self.data_S = np.stack(data_S)
        self.data_A = np.stack(data_A)
        self.data_X = np.stack(data_X)
        self.data_D = np.stack(data_D)
        self.data_F = np.stack(data_F)
        self.data_Le = np.stack(data_Le)
        self.data_Lv = np.stack(data_Lv)
        self.__len = len(self.data)

    def _genA(self, mol: Mol, connected: bool = True, max_length: Optional[int] = None) -> Optional[NDArray]:
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        for b in mol.GetBonds():
            begin = b.GetBeginAtomIdx()
            end = b.GetEndAtomIdx()
            bond_type = self.bond_encoder_m[b.GetBondType()]

            A[begin, end] = bond_type
            A[end, begin] = bond_type

        degree = np.sum(A[: mol.GetNumAtoms(), : mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol: Mol, max_length: Optional[int] = None) -> NDArray:
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array(
            [self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
            + [0] * (max_length - mol.GetNumAtoms()),
            dtype=np.int32,
        )

    def _genS(self, mol: Mol, max_length: Optional[int] = None) -> NDArray:
        max_length = max_length if max_length is not None else len(Chem.MolToSmiles(mol))

        return np.array(
            [self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)]
            + [self.smiles_encoder_m["E"]] * (max_length - len(Chem.MolToSmiles(mol))),
            dtype=np.int32,
        )

    def _genF(self, mol: Mol, max_length: Optional[int] = None) -> NDArray:
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array(
            [
                [
                    *[a.GetDegree() == i for i in range(5)],
                    *[a.GetExplicitValence() == i for i in range(9)],
                    *[int(a.GetHybridization()) == i for i in range(1, 7)],
                    *[a.GetImplicitValence() == i for i in range(9)],
                    a.GetIsAromatic(),
                    a.GetNoImplicit(),
                    *[a.GetNumExplicitHs() == i for i in range(5)],
                    *[a.GetNumImplicitHs() == i for i in range(5)],
                    *[a.GetNumRadicalElectrons() == i for i in range(5)],
                    a.IsInRing(),
                    *[a.IsInRingSize(i) for i in range(2, 9)],
                ]
                for a in mol.GetAtoms()
            ],
            dtype=np.int32,
        )

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def matrices2mol(self, node_labels: NDArray, edge_labels: NDArray, strict: bool = False) -> Mol:
        mol: Mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol = None

        return mol

    def seq2mol(self, seq: list[int], strict: bool = False) -> Optional[Mol]:
        mol = Chem.MolFromSmiles("".join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol = None

        return mol

    def _generate_train_validation_test(self, validation: float, test: float) -> None:
        self.log("Creating train, validation and test sets..")

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train : train + validation]
        self.test_idx = self.all_idx[train + validation :]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        self.log(f"Created train ({train} items), validation ({validation} items) and test ({test} items) sets!")

    def _next_batch(
        self, counter: int, count: int, idx: NDArray[np.int_], batch_size: Optional[int]
    ) -> tuple[int, list[NDArray]]:
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [
                obj[idx[counter : counter + batch_size]]
                for obj in (
                    self.data,
                    self.smiles,
                    self.data_S,
                    self.data_A,
                    self.data_X,
                    self.data_D,
                    self.data_F,
                    self.data_Le,
                    self.data_Lv,
                )
            ]

            counter += batch_size
        else:
            output = [
                obj[idx]
                for obj in (
                    self.data,
                    self.smiles,
                    self.data_S,
                    self.data_A,
                    self.data_X,
                    self.data_D,
                    self.data_F,
                    self.data_Le,
                    self.data_Lv,
                )
            ]

        return counter, output

    def next_train_batch(self, batch_size: Optional[int] = None) -> list[NDArray]:
        train_counter, out = self._next_batch(
            counter=self.train_counter,
            count=self.train_count,
            idx=self.train_idx,
            batch_size=batch_size,
        )
        self.train_counter = train_counter

        return out

    def next_validation_batch(self, batch_size: Optional[int] = None) -> list[NDArray]:
        validation_counter, out = self._next_batch(
            counter=self.validation_counter,
            count=self.validation_count,
            idx=self.validation_idx,
            batch_size=batch_size,
        )
        self.validation_counter = validation_counter

        return out

    def next_test_batch(self, batch_size: Optional[int] = None) -> list[NDArray]:
        counter, out = self._next_batch(
            counter=self.test_counter,
            count=self.test_count,
            idx=self.test_idx,
            batch_size=batch_size,
        )
        self.test_counter = counter

        return out

    @staticmethod
    def log(msg: str = "", date: bool = True) -> None:
        print(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + " " + str(msg) if date else str(msg))

    def __len__(self) -> int:
        return self.__len


if __name__ == "__main__":
    # data = SparseMolecularDataset()
    # data.generate('gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('gdb9_9nodes.sparsedataset')

    data = SparseMolecularDataset()
    data.generate("data/qm9_5k.smi", validation=0.0021, test=0.0021)  # , filters=lambda x: x.GetNumAtoms() <= 9)
    data.save("data/qm9_5k.sparsedataset")
