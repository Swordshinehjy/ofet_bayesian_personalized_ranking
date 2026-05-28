"""Chemistry preprocessing: cyclization, extra feature extraction, DataFrame preprocessing."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from .config import EXTRA_COLS, TASK_NAMES

logger = logging.getLogger(__name__)


def cyclize_polymer_with_cp_marking(
        smiles: str) -> Tuple[Optional[str], Optional[Chem.Mol]]:
    """
    Cyclize polymer and mark connection points (CP).

    Returns
    -------
    Tuple[smiles, mol]
        - smiles: Cyclized SMILES string
        - mol: Cyclized RDKit Mol object with connection point atoms marked with is_cp property
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    Chem.SanitizeMol(mol)

    dummy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*'
    ]
    if len(dummy_indices) != 2:
        return None, None
    try:
        idx1, idx2 = dummy_indices[0], dummy_indices[1]
        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)
        neighbors1 = atom1.GetNeighbors()
        neighbors2 = atom2.GetNeighbors()
        if not neighbors1 or not neighbors2:
            return None, None
        n1_idx = neighbors1[0].GetIdx()
        n2_idx = neighbors2[0].GetIdx()

        bond1 = mol.GetBondBetweenAtoms(idx1, n1_idx)
        bond2 = mol.GetBondBetweenAtoms(idx2, n2_idx)
        bond_type = Chem.BondType.SINGLE
        if bond1 is not None and bond1.GetBondType() != Chem.BondType.SINGLE:
            bond_type = bond1.GetBondType()
        elif bond2 is not None:
            bond_type = bond2.GetBondType()
        bond_dir = Chem.BondDir.NONE
        if bond1 is not None and bond1.GetBondDir() != Chem.BondDir.NONE:
            bond_dir = bond1.GetBondDir()
        elif bond2 is not None:
            bond_dir = bond2.GetBondDir()

        rw_mol = Chem.RWMol(mol)

        # If n1 == n2 (shared neighbor), just remove dummies — no new bond needed
        if n1_idx != n2_idx:
            existing_bond = rw_mol.GetBondBetweenAtoms(n1_idx, n2_idx)
            if existing_bond is None:
                rw_mol.AddBond(n1_idx, n2_idx, order=bond_type)
                new_bond = rw_mol.GetBondBetweenAtoms(n1_idx, n2_idx)
                if bond_dir != Chem.BondDir.NONE and new_bond is not None:
                    new_bond.SetBondDir(bond_dir)

        cp_atom_indices = [n1_idx, n2_idx]

        for idx in sorted(dummy_indices, reverse=True):
            if idx < n1_idx:
                n1_idx -= 1
            if idx < n2_idx:
                n2_idx -= 1
            rw_mol.RemoveAtom(idx)

        cp_atom_indices = [n1_idx, n2_idx]

        cyclic_mol = rw_mol.GetMol()
        Chem.SanitizeMol(cyclic_mol)
        Chem.AssignStereochemistry(cyclic_mol, force=True, cleanIt=True)

        for cp_idx in cp_atom_indices:
            atom = cyclic_mol.GetAtomWithIdx(cp_idx)
            atom.SetBoolProp("is_cp", True)

        return Chem.MolToSmiles(cyclic_mol), cyclic_mol

    except Exception as e:
        logger.error(f"Error processing smiles {smiles}: {e}")
        return None, None


def cyclize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Perform cyclization on DataFrame, add cyc_1/2, mol_1/2 columns."""
    df = df.copy()
    for s in ["1", "2"]:
        results = df[f"Polymer_{s}"].apply(cyclize_polymer_with_cp_marking)
        df[f"cyc_{s}"] = results.apply(lambda x: x[0])
        df[f"mol_{s}"] = results.apply(lambda x: x[1])
    return df


def extra_feat(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """Extract extra feature columns, fill NaN with 0."""
    cols = [c.format(s=suffix) for c in EXTRA_COLS]
    return df[cols].fillna(0.0).values.astype(np.float32)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Read CSV and perform:
      - Polymer cyclization
      - Take log10 of mobility columns
      - Drop invalid rows
    """
    df = pd.read_csv(csv_path)
    df = cyclize_df(df)

    target_raw = [f"{t}_{s}" for t in TASK_NAMES for s in ("1", "2")]
    df = df.dropna(subset=["cyc_1", "cyc_2"] +
                   target_raw).reset_index(drop=True)
    df = df[df["mol_1"].notna() & df["mol_2"].notna()].reset_index(drop=True)

    for col in target_raw:
        df[f"log_{col}"] = np.log10(df[col].clip(lower=1e-12))

    logger.info(f"Preprocessed dataset: {len(df)} valid pairs")
    return df
