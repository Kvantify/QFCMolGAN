import gzip
import math
import pickle
from typing import Callable, Optional, Any, TypeAlias, TypeGuard

import numpy as np
import rdkit
from numpy.typing import ArrayLike, NDArray
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from rdkit.Chem import rdchem

from data.sparse_molecular_dataset import SparseMolecularDataset
from utils.utils_io import random_string

NP_model = pickle.load(gzip.open("data/NP_score.pkl.gz"))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open("data/SA_score.pkl.gz")) for j in range(1, len(i))}

Mol: TypeAlias = rdchem.Mol


class MolecularMetrics:
    @staticmethod
    def _avoid_sanitization_error(op: Callable[[], Any]) -> Optional[Any]:
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x: ArrayLike, x_min: ArrayLike, x_max: ArrayLike) -> ArrayLike:
        if not (
            type(x) is type(x_min) is type(x_max)
            or (isinstance(x_min, (int, float)) and isinstance(x_min, (int, float)))
        ):
            raise TypeError("Both arrays must have the same data type.")
        return (x - x_min) / (x_max - x_min)  # type: ignore

    @staticmethod
    def valid_lambda(x: Optional[Mol]) -> TypeGuard[bool]:
        return x is not None and Chem.MolToSmiles(x) != ""

    @staticmethod
    def valid_lambda_special(x: Optional[Mol]) -> bool:
        s = Chem.MolToSmiles(x) if x is not None else ""
        return "*" not in s and "." not in s and s != ""

    @staticmethod
    def valid_scores(mols: list[Mol]) -> NDArray:
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols: list[Mol]) -> list[bool]:
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols: list[Mol]) -> float:
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean().item()

    @staticmethod
    def novel_scores(mols: list[Mol], data: SparseMolecularDataset) -> NDArray:
        return np.array(
            list(
                map(
                    lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles,
                    mols,
                )
            )
        )

    @staticmethod
    def novel_filter(mols: list[Mol], data: SparseMolecularDataset) -> list[filter]:
        return list(
            filter(
                lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles,  # type: ignore
                mols,
            )
        )

    @staticmethod
    def novel_total_score(mols: list[Mol], data: SparseMolecularDataset) -> float:
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()

    @staticmethod
    def unique_scores(mols: list[Mol]) -> NDArray:
        smiles: list[str] = list(
            map(
                lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else "",
                mols,
            )
        )
        return np.clip(
            0  # was previously 0.75
            + np.array(
                list(map(lambda x: 1 / smiles.count(x) if x != "" else 0, smiles)),
                dtype=np.float32,
            ),
            0,
            1,
        )

    @staticmethod
    def unique_total_score(mols: list[Mol]) -> float:
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    @staticmethod
    def natural_product_scores(mols: list[Mol], norm: bool = False) -> NDArray:
        # calculating the score
        scores = [
            (
                sum(
                    NP_model.get(bit, 0)
                    for bit in Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2).GetNonzeroElements()
                )
                / float(mol.GetNumAtoms())  # type: ignore
                if mol is not None
                else None
            )
            for mol in mols
        ]

        # preventing score explosion for exotic molecules
        scores = list(
            map(
                lambda score: (
                    score
                    if score is None
                    else (
                        4 + math.log10(score - 4 + 1)
                        if score > 4
                        else (-4 - math.log10(-4 - score + 1) if score < -4 else score)
                    )
                ),
                scores,
            )
        )

        scores_arr = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores_arr = np.clip(MolecularMetrics.remap(scores_arr, -3, 1), 0.0, 1.0) if norm else scores_arr

        return scores_arr

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols: list[Mol], norm: bool = False) -> NDArray:
        return np.array(
            list(
                map(
                    lambda x: 0 if x is None else x,
                    [
                        MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None
                        for mol in mols
                    ],
                )
            )
        )

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols: list[Mol], norm: bool = False) -> NDArray:
        scores = [
            MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
            for mol in mols
        ]
        scores_arr = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores_arr = (
            np.clip(MolecularMetrics.remap(scores_arr, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores_arr
        )

        return scores_arr

    @staticmethod
    def _compute_SAS(mol: Mol) -> float:
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.0
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * 0.5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
        # smooth the 10-end
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols: list[Mol], norm: bool = False) -> NDArray:
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in mols]
        scores_arr = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores_arr = np.clip(MolecularMetrics.remap(scores_arr, 5, 1.5), 0.0, 1.0) if norm else scores_arr

        return scores_arr

    @staticmethod
    def diversity_scores(mols: list[Mol], data: SparseMolecularDataset) -> NDArray:
        rand_mols = np.random.choice(data.data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(
                map(
                    lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0,
                    mols,
                )
            )
        )
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol: Mol, fps: Any) -> float:
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist).item()
        return score

    @staticmethod
    def drugcandidate_scores(mols: list[Mol], data: SparseMolecularDataset) -> NDArray:
        scores = (
            MolecularMetrics.constant_bump(
                MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True),
                0.210,
                0.945,
            )
            + MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            + MolecularMetrics.novel_scores(mols, data)
            + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3
        ) / 4

        return scores

    @staticmethod
    def constant_bump(x: NDArray, x_low: ArrayLike, x_high: ArrayLike, decay: float = 0.025) -> NDArray:
        return np.select(
            condlist=[x <= x_low, x >= x_high],
            choicelist=[
                np.exp(-((x - x_low) ** 2) / decay),
                np.exp(-((x - x_high) ** 2) / decay),
            ],
            default=np.ones_like(x),
        )


def all_scores(
    mols: list[Mol], data: SparseMolecularDataset, norm: bool = False, reconstruction: bool = False
) -> tuple[dict[str, NDArray], dict[str, float]]:
    m0 = {
        "NP": MolecularMetrics.natural_product_scores(mols, norm=norm),
        "QED": MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        "Solute": MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        "SA": MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
        "diverse": MolecularMetrics.diversity_scores(mols, data),
        "drugcand": MolecularMetrics.drugcandidate_scores(mols, data),
    }

    m1 = {
        "valid": MolecularMetrics.valid_total_score(mols) * 100,
        "unique": MolecularMetrics.unique_total_score(mols) * 100,
        "novel": MolecularMetrics.novel_total_score(mols, data) * 100,
    }

    return m0, m1


def save_mol_img(mols: list[Mol] | NDArray[Mol], f_name: str = "tmp.png", is_test: bool = False) -> None:
    print("Generating molecules...")
    orig_f_name = f_name
    for a_mol in mols:
        try:
            if Chem.MolToSmiles(a_mol) is not None:
                if is_test:
                    f_name = orig_f_name
                    f_split = f_name.split(".")
                    f_split[-1] = random_string() + "." + f_split[-1]
                    f_name = "".join(f_split)

                rdkit.Chem.Draw.MolToFile(a_mol, f_name)
                # a_smi = Chem.MolToSmiles(a_mol)
                # mol_graph = read_smiles(a_smi)

                # break only give you one image
                # break

                # if not is_test:
                #     break
        except Exception:
            continue
