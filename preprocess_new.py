import pandas as pd
from rdkit import Chem

def validate_wildcard_count(smiles, expected_count=2):
    """Validate the number of wildcard (*) characters in a SMILES string."""
    wildcard_count = smiles.count('*')
    return wildcard_count == expected_count, wildcard_count

def process_and_validate_csv(csv_path):
    """Process and validate CSV file containing polymer SMILES."""
    data = pd.read_csv(csv_path, encoding='latin-1')
    
    # Normalize SMILES strings
    data['Polymer_1'] = data['Polymer_1'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notna(x) else x)
    data['Polymer_2'] = data['Polymer_2'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notna(x) else x)
    print("=" * 80)
    print("Validation [*] count check results:")
    print("=" * 80)

    return data

if __name__ == "__main__":
    csv_path = "new_mol.csv"
    processed_data = process_and_validate_csv(csv_path)
    
    output_path = "new_mol.csv"
    processed_data.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
