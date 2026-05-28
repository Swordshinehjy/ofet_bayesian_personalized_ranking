import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_polymer_smiles():
    return "*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"


@pytest.fixture
def sample_invalid_smiles():
    return "not_a_valid_smiles_xyz"


@pytest.fixture
def sample_smiles_no_wildcard():
    return "c1ccccc1"


@pytest.fixture
def sample_smiles_one_wildcard():
    return "*c1ccccc1"


@pytest.fixture
def sample_df():
    import pandas as pd
    return pd.DataFrame({
        "Materials_1": ["PolyA"],
        "Polymer_1": ["*c1cc2c(s1)-c1cc3c(cc1[Si]2(C)C)-c1sc(-c2ccc(*)c4nsnc24)cc1[Si]3(C)C"],
        "conjugation_1": [1],
        "Isomer_1": [0.0],
        "CentroSymmetry_1": [0],
        "mu_e_1": [0.014],
        "mu_h_1": [0.01],
        "E_LUMO (eV)_1": [-3.1],
        "E_HOMO (eV)_1": [-5.3],
        "Materials_2": ["PolyB"],
        "Polymer_2": ["*c1ccc(-c2ccc(-c3ccc(-c4cc5c(s4)-c4cc6c(cc4[Si]5(C)C)-c4sc(*)cc4[Si]6(C)C)s3)c3nsnc23)s1"],
        "conjugation_2": [1],
        "Isomer_2": [0.0],
        "CentroSymmetry_2": [0],
        "mu_e_2": [0.28],
        "mu_h_2": [0.05],
        "E_LUMO (eV)_2": [-3.4],
        "E_HOMO (eV)_2": [-5.2],
    })
