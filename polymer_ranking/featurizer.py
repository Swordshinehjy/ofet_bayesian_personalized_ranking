"""Featurizer factory and custom atom featurizer."""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType

try:
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.featurizers.atom import MultiHotAtomFeaturizer
    from chemprop.featurizers.bond import MultiHotBondFeaturizer
    from chemprop.conf import DEFAULT_BOND_FDIM
except ImportError:
    pass

from .config import CP_FEATURE_DIM


class CustomMultiHotAtomFeaturizer(MultiHotAtomFeaturizer):
    """
    chemprop MultiHotAtomFeaturizer + 1 connection point (CP) binary feature.

    Atom feature vector = chemprop default features || [is_cp]
    Requires atom.SetBoolProp("is_cp", True/False) to be set beforehand.
    """

    def __init__(self, atomic_nums=None):
        if atomic_nums is None:
            atomic_nums = list(range(1, 37)) + [52, 53]

        super().__init__(
            atomic_nums=atomic_nums,
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
        )

    def __call__(self, atom: Chem.Atom) -> np.ndarray:
        base = super().__call__(atom)
        is_cp = float(atom.HasProp("is_cp") and atom.GetBoolProp("is_cp"))
        return np.append(base, is_cp).astype(np.float32)

    def __len__(self) -> int:
        return super().__len__() + 1


def create_featurizer() -> SimpleMoleculeMolGraphFeaturizer:
    """Create unified molecule graph featurizer (eliminates duplicate creation)."""
    return SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=CustomMultiHotAtomFeaturizer(),
        bond_featurizer=MultiHotBondFeaturizer(),
    )


def get_feature_dims():
    """Return (atom_fdim, bond_fdim) based on CustomMultiHotAtomFeaturizer."""
    atom_fdim = len(CustomMultiHotAtomFeaturizer())
    bond_fdim = DEFAULT_BOND_FDIM
    return atom_fdim, bond_fdim
