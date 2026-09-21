import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from polymer_ranking.chemistry import (
    cyclize_polymer_with_cp_marking,
    cyclize_df,
    extra_feat,
    get_attachment_atoms,
    get_cp_atoms,
    attachment_topological_distance,
    cp_topological_distance,
    build_linear_oligomer,
    choose_oligomer_repeats,
)
from polymer_ranking.config import min_span_for_depth


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


class TestGetAttachmentAtoms:
    def test_two_attachment_atoms(self):
        # *-CH2-CH2-*
        pair = get_attachment_atoms("*CC(*)")
        assert pair is not None
        i, j = pair
        mol = Chem.MolFromSmiles("*CC(*)")
        assert mol.GetAtomWithIdx(i).GetSymbol() == "C"
        assert mol.GetAtomWithIdx(j).GetSymbol() == "C"
        assert i != j

    def test_no_attachment_point(self):
        assert get_attachment_atoms("c1ccccc1") is None

    def test_already_cyclized_mol(self, sample_polymer_smiles):
        _, mol = cyclize_polymer_with_cp_marking(sample_polymer_smiles)
        assert get_attachment_atoms(mol) is None

    @pytest.mark.parametrize("smiles", ["*c1ccccc1", "*C*C*C", "not_a_smiles"])
    def test_invalid_attachment_counts(self, smiles):
        assert get_attachment_atoms(smiles) is None

    def test_accepts_mol_object(self):
        mol = Chem.MolFromSmiles("*c1ccc(*)cc1")
        assert get_attachment_atoms(mol) == get_attachment_atoms("*c1ccc(*)cc1")


class TestAttachmentTopologicalDistance:
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("*CC(*)", 1),          # *-CH2-CH2-*
            ("*CCC(*)", 2),         # *-CH2-CH2-CH2-*
            ("*c1ccc(*)cc1", 3),    # para substituted benzene
            ("*c1ccsc1*", 1),       # thiophene: attachment carbons are bonded
        ],
    )
    def test_known_distances(self, smiles, expected):
        assert attachment_topological_distance(smiles) == expected

    def test_shared_attachment_atom(self):
        # both attachment points on the same carbon -> zero span
        assert attachment_topological_distance("*C(*)") == 0

    def test_distance_on_fixture_unit(self, sample_polymer_smiles):
        dist = attachment_topological_distance(sample_polymer_smiles)
        assert isinstance(dist, int)
        assert dist > 0

    def test_cyclization_collapses_span_to_one(self, sample_polymer_smiles):
        dist = attachment_topological_distance(sample_polymer_smiles)
        _, mol = cyclize_polymer_with_cp_marking(sample_polymer_smiles)
        cp_atoms = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.HasProp("is_cp") and atom.GetBoolProp("is_cp")
        ]
        assert len(cp_atoms) == 2
        assert len(Chem.GetShortestPath(mol, cp_atoms[0], cp_atoms[1])) - 1 == 1
        assert dist > 1

    def test_accepts_mol_object(self):
        mol = Chem.MolFromSmiles("*c1ccc(*)cc1")
        assert attachment_topological_distance(mol) == 3

    @pytest.mark.parametrize(
        "smiles", ["not_a_valid_smiles_xyz", "c1ccccc1", "*c1ccccc1", "*C*C*C"]
    )
    def test_invalid_inputs_return_none(self, smiles):
        assert attachment_topological_distance(smiles) is None

    def test_depth_requirement_check(self):
        """L >= 2 * depth + 1 is the criterion for a safe cyclization."""
        depth = 6
        # thiophene-like unit: far too short, needs oligomer expansion
        assert attachment_topological_distance("*c1ccsc1*") < 2 * depth + 1


class TestBuildLinearOligomer:
    def test_single_unit_strips_dummies(self):
        mol = build_linear_oligomer("*c1ccsc1*", 1)
        assert mol is not None
        assert mol.GetNumAtoms() == 5  # thiophene without the two dummies
        assert Chem.MolToSmiles(mol) == "c1ccsc1"

    def test_oligomer_size_and_span(self):
        n = 3
        mol = build_linear_oligomer("*c1ccsc1*", n)
        assert mol.GetNumAtoms() == 5 * n
        # span of an n-mer = n * L + (n - 1)
        assert cp_topological_distance(mol) == n * 1 + (n - 1)

    def test_open_ends_marked_as_cp(self):
        mol = build_linear_oligomer("*c1ccc(*)cc1", 3)
        cp = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.HasProp("is_cp") and a.GetBoolProp("is_cp")
        ]
        assert len(cp) == 2
        assert cp_topological_distance(mol) == 3 * 3 + 2

    def test_oligomer_is_sane(self):
        mol = build_linear_oligomer("*c1ccsc1*", 4)
        assert Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE

    @pytest.mark.parametrize("smiles,n", [("not_a_smiles", 2), ("c1ccccc1", 2),
                                          ("*c1ccsc1*", 0)])
    def test_invalid_inputs(self, smiles, n):
        assert build_linear_oligomer(smiles, n) is None


class TestChooseOligomerRepeats:
    def test_long_unit_not_expanded(self, sample_polymer_smiles):
        assert attachment_topological_distance(
            sample_polymer_smiles) >= min_span_for_depth()
        assert choose_oligomer_repeats(sample_polymer_smiles) == 1

    @pytest.mark.parametrize("smiles,expected", [("*c1ccsc1*", 7),
                                                 ("*c1ccc(*)cc1", 4),
                                                 ("*CC(*)", 7)])
    def test_short_units(self, smiles, expected):
        assert choose_oligomer_repeats(smiles) == expected

    def test_depth_drives_requirement(self):
        # depth=2 -> required span 5 -> 3 thiophene units (3 * 1 + 2)
        assert choose_oligomer_repeats("*c1ccsc1*", depth=2) == 3
        assert choose_oligomer_repeats("*c1ccsc1*", min_span=5) == 3
        assert choose_oligomer_repeats("*c1ccsc1*", depth=0) == 1

    def test_shared_attachment_not_expanded(self):
        assert choose_oligomer_repeats("*C*") == 1

    def test_max_repeats_cap(self):
        assert choose_oligomer_repeats("*c1ccsc1*", max_repeats=2) == 2

    def test_invalid_input(self):
        assert choose_oligomer_repeats("not_a_smiles") is None
        assert choose_oligomer_repeats("c1ccccc1") is None

    def test_chosen_repeats_reach_required_span(self):
        for smiles in ["*c1ccsc1*", "*c1ccc(*)cc1", "*CC(*)"]:
            n = choose_oligomer_repeats(smiles)
            linear = build_linear_oligomer(smiles, n)
            assert cp_topological_distance(linear) >= min_span_for_depth()


class TestCyclizeWithOligomerExpansion:
    def test_short_unit_is_expanded(self):
        cyc, mol = cyclize_polymer_with_cp_marking("*c1ccsc1*")
        assert mol.GetIntProp("n_repeat") == 7
        assert mol.GetNumAtoms() == 5 * 7
        assert "*" not in cyc
        pair = get_cp_atoms(mol)
        # ring closure: the two ends are one bond apart
        assert len(Chem.GetShortestPath(mol, pair[0], pair[1])) - 1 == 1
        assert Chem.SanitizeMol(mol,
                                catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE

    def test_auto_oligomer_can_be_disabled(self):
        _, mol = cyclize_polymer_with_cp_marking("*c1ccsc1*",
                                                 auto_oligomer=False)
        assert mol.GetIntProp("n_repeat") == 1
        assert mol.GetNumAtoms() == 5

    def test_long_unit_unchanged(self, sample_polymer_smiles):
        _, mol = cyclize_polymer_with_cp_marking(sample_polymer_smiles)
        assert mol.GetIntProp("n_repeat") == 1

    def test_depth_controls_expansion(self):
        _, mol = cyclize_polymer_with_cp_marking("*c1ccsc1*", depth=2)
        assert mol.GetIntProp("n_repeat") == 3

    def test_max_repeats_respected(self):
        _, mol = cyclize_polymer_with_cp_marking("*c1ccsc1*", max_repeats=2)
        assert mol.GetIntProp("n_repeat") == 2

    def test_cp_atoms_marked_after_expansion(self):
        _, mol = cyclize_polymer_with_cp_marking("*c1ccc(*)cc1")
        cp = [
            a for a in mol.GetAtoms()
            if a.HasProp("is_cp") and a.GetBoolProp("is_cp")
        ]
        assert len(cp) == 2

    def test_invalid_smiles_still_none(self):
        assert cyclize_polymer_with_cp_marking("not_a_smiles") == (None, None)
        assert cyclize_polymer_with_cp_marking("*C*C*") == (None, None)


class TestCyclizeDfOligomer:
    def test_cyclize_df_expands_short_units(self):
        df = pd.DataFrame({
            "Polymer_1": ["*c1ccsc1*", "*CC(*)"],
            "Polymer_2": ["*c1ccc(*)cc1", "*CC(*)"],
        })
        result = cyclize_df(df)
        assert result["mol_1"].notna().all()
        assert [m.GetIntProp("n_repeat") for m in result["mol_1"]] == [7, 7]
        assert [m.GetIntProp("n_repeat") for m in result["mol_2"]] == [4, 7]

    def test_cyclize_df_legacy_mode(self):
        df = pd.DataFrame({
            "Polymer_1": ["*c1ccsc1*"],
            "Polymer_2": ["*CC(*)"],
        })
        result = cyclize_df(df, auto_oligomer=False)
        assert all(m.GetIntProp("n_repeat") == 1 for m in result["mol_1"])
        assert all(m.GetIntProp("n_repeat") == 1 for m in result["mol_2"])
