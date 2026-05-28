import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from polymer_ranking.chemistry import (
    cyclize_polymer_with_cp_marking,
    cyclize_df,
    extra_feat,
)


class TestCyclizePolymerWithCpMarking:
    def test_valid_polymer_smiles(self, sample_polymer_smiles):
        smiles, mol = cyclize_polymer_with_cp_marking(sample_polymer_smiles)
        assert smiles is not None
        assert mol is not None
        assert isinstance(smiles, str)
        assert isinstance(mol, Chem.Mol)
        assert "*" not in smiles

    def test_invalid_smiles(self, sample_invalid_smiles):
        smiles, mol = cyclize_polymer_with_cp_marking(sample_invalid_smiles)
        assert smiles is None
        assert mol is None

    def test_no_wildcard(self, sample_smiles_no_wildcard):
        smiles, mol = cyclize_polymer_with_cp_marking(sample_smiles_no_wildcard)
        assert smiles is None
        assert mol is None

    def test_one_wildcard(self, sample_smiles_one_wildcard):
        smiles, mol = cyclize_polymer_with_cp_marking(sample_smiles_one_wildcard)
        assert smiles is None
        assert mol is None

    def test_cp_atoms_marked(self, sample_polymer_smiles):
        _, mol = cyclize_polymer_with_cp_marking(sample_polymer_smiles)
        cp_atoms = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.HasProp("is_cp") and atom.GetBoolProp("is_cp")
        ]
        assert len(cp_atoms) == 2

    def test_cyclized_mol_is_sane(self, sample_polymer_smiles):
        _, mol = cyclize_polymer_with_cp_marking(sample_polymer_smiles)
        assert mol.GetNumAtoms() > 0
        errors = Chem.SanitizeMol(mol, catchErrors=True)
        assert errors == Chem.SanitizeFlags.SANITIZE_NONE

    def test_returns_tuple(self, sample_polymer_smiles):
        result = cyclize_polymer_with_cp_marking(sample_polymer_smiles)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_three_wildcards(self):
        smiles = "*C*C*C"
        result_smiles, result_mol = cyclize_polymer_with_cp_marking(smiles)
        assert result_smiles is None
        assert result_mol is None


class TestCyclizeDf:
    def test_cyclize_df_adds_columns(self, sample_df):
        result = cyclize_df(sample_df)
        assert "cyc_1" in result.columns
        assert "cyc_2" in result.columns
        assert "mol_1" in result.columns
        assert "mol_2" in result.columns

    def test_cyclize_df_does_not_modify_original(self, sample_df):
        original_cols = set(sample_df.columns)
        result = cyclize_df(sample_df)
        assert set(sample_df.columns) == original_cols
        assert "cyc_1" not in sample_df.columns

    def test_cyclize_df_valid_smiles(self, sample_df):
        result = cyclize_df(sample_df)
        assert result["cyc_1"].notna().all()
        assert result["cyc_2"].notna().all()
        assert result["mol_1"].notna().all()
        assert result["mol_2"].notna().all()

    def test_cyclize_df_invalid_smiles(self):
        df = pd.DataFrame({
            "Materials_1": ["X"],
            "Polymer_1": ["invalid_smiles"],
            "Materials_2": ["Y"],
            "Polymer_2": ["invalid_smiles"],
        })
        result = cyclize_df(df)
        assert result["cyc_1"].isna().all()
        assert result["cyc_2"].isna().all()


class TestExtraFeat:
    def test_extra_feat_shape(self, sample_df):
        df = cyclize_df(sample_df)
        feat = extra_feat(df, "1")
        assert feat.shape == (1, 5)

    def test_extra_feat_dtype(self, sample_df):
        df = cyclize_df(sample_df)
        feat = extra_feat(df, "1")
        assert feat.dtype == np.float32

    def test_extra_feat_fills_nan(self):
        df = pd.DataFrame({
            "conjugation_1": [1],
            "Isomer_1": [np.nan],
            "CentroSymmetry_1": [0],
            "E_LUMO (eV)_1": [-3.1],
            "E_HOMO (eV)_1": [-5.3],
        })
        feat = extra_feat(df, "1")
        assert not np.isnan(feat).any()
        assert feat[0, 1] == 0.0

    def test_extra_feat_suffix_2(self, sample_df):
        df = cyclize_df(sample_df)
        feat1 = extra_feat(df, "1")
        feat2 = extra_feat(df, "2")
        assert feat1.shape == feat2.shape
