"""Extract the oligomer-generation algorithm from polymer_ranking/chemistry.py.

Builds linear head-to-tail oligomers from polymer repeat units (SMILES with
two ``*`` attachment points) but skips the final cyclization step
(``_cyclize_oligomer`` / ``_close_ring`` in chemistry.py), so the open-chain
oligomer can be inspected against the original repeat unit.

Output CSV columns:
  polymer            original repeat unit SMILES
  span_L             backbone span (bonds) between the two attachment atoms
  n_repeats          number of repeat units chained (chosen by the algorithm)
  oligomer           generated linear oligomer SMILES (NO ring closure)
  oligomer_cp_mapped oligomer SMILES with the two open ends mapped as [1]/[2]
  cp_span_measured   span between the open ends in the built oligomer
  cp_span_expected   theoretical n * L + (n - 1)
  span_ok            measured span >= 2 * depth + 1 (D-MPNN receptive field)
  status             ok / error description
"""

import argparse
import math
from collections import deque
from typing import FrozenSet, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem

# Defaults mirrored from polymer_ranking/config.py
DEFAULT_DEPTH = 6
OLIGOMER_MAX_REPEATS = 8


def min_span_for_depth(depth: int = DEFAULT_DEPTH) -> int:
    """Minimum backbone span required: 2 * depth + 1 bonds."""
    return 2 * depth + 1


MolLike = Union[str, Chem.Mol]


def _as_mol(smiles_or_mol: MolLike) -> Optional[Chem.Mol]:
    """Accept a SMILES string or an RDKit Mol and always return a Mol (None if invalid)."""
    if isinstance(smiles_or_mol, Chem.Mol):
        return smiles_or_mol
    if not isinstance(smiles_or_mol, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles_or_mol)
    except Exception:
        return None


def get_attachment_atoms(smiles_or_mol: MolLike) -> Optional[Tuple[int, int]]:
    """Locate the two heavy atoms bonded to the two ``*`` dummy atoms."""
    mol = _as_mol(smiles_or_mol)
    if mol is None:
        return None

    dummy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"
    ]
    if len(dummy_indices) != 2:
        return None

    attachment_indices = []
    for idx in dummy_indices:
        neighbors = [
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nb.GetSymbol() != "*"
        ]
        if not neighbors:
            return None
        attachment_indices.append(neighbors[0])

    return attachment_indices[0], attachment_indices[1]


def _shortest_bond_distance(
        mol: Chem.Mol,
        start: int,
        end: int,
        blocked: FrozenSet[int] = frozenset()) -> Optional[int]:
    """Number of bonds on the shortest path between two atoms (BFS)."""
    if start == end:
        return 0

    dist = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == end:
            return dist[current]
        for nb in mol.GetAtomWithIdx(current).GetNeighbors():
            n_idx = nb.GetIdx()
            if n_idx in blocked or n_idx in dist:
                continue
            dist[n_idx] = dist[current] + 1
            queue.append(n_idx)

    return None


def attachment_topological_distance(smiles_or_mol: MolLike) -> Optional[int]:
    """Span L (in bonds) between the two attachment atoms of a repeat unit."""
    mol = _as_mol(smiles_or_mol)
    if mol is None:
        return None

    pair = get_attachment_atoms(mol)
    if pair is None:
        return None
    start, end = pair
    if start == end:
        return 0

    blocked = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                        if atom.GetSymbol() == "*")
    return _shortest_bond_distance(mol, start, end, blocked)


def get_cp_atoms(mol: Optional[Chem.Mol]) -> Optional[Tuple[int, int]]:
    """Return the two atoms marked with ``is_cp`` (the open ends)."""
    if mol is None:
        return None
    indices = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.HasProp("is_cp") and atom.GetBoolProp("is_cp")
    ]
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0], indices[0]
    return indices[0], indices[1]


def cp_topological_distance(mol: Optional[Chem.Mol]) -> Optional[int]:
    """Span between the two ``is_cp`` atoms of a linear oligomer (in bonds)."""
    pair = get_cp_atoms(mol)
    if pair is None or mol is None:
        return None
    return _shortest_bond_distance(mol, pair[0], pair[1])


def _link_bond_info(
        mol: Chem.Mol,
        dummy_a: int,
        attach_a: int,
        dummy_b: int,
        attach_b: int) -> Tuple[Chem.BondType, Chem.BondDir]:
    """Bond type / direction used for head-to-tail links between units."""
    bonds = [
        mol.GetBondBetweenAtoms(dummy_a, attach_a),
        mol.GetBondBetweenAtoms(dummy_b, attach_b),
    ]
    bond_type = Chem.BondType.SINGLE
    for bond in bonds:
        if bond is not None and bond.GetBondType() != Chem.BondType.SINGLE:
            bond_type = bond.GetBondType()
            break
    bond_dir = Chem.BondDir.NONE
    for bond in bonds:
        if bond is not None and bond.GetBondDir() != Chem.BondDir.NONE:
            bond_dir = bond.GetBondDir()
            break
    return bond_type, bond_dir


def _shift_index(idx: int, removed) -> int:
    """Map an atom index to its value after the indices in ``removed`` are gone."""
    return idx - sum(1 for r in removed if r < idx)


def build_linear_oligomer(smiles_or_mol: MolLike,
                          n: int = 1) -> Optional[Chem.Mol]:
    """Chain ``n`` copies of a repeat unit head-to-tail and strip the ``*`` dummies.

    Extracted verbatim from chemistry.py. The result stays OPEN: head of the
    first copy and tail of the last copy are marked with ``is_cp``. The
    cyclization step that chemistry.py applies afterwards is intentionally
    NOT performed here.
    """
    mol = _as_mol(smiles_or_mol)
    if mol is None or n < 1:
        return None

    pair = get_attachment_atoms(mol)
    if pair is None:
        return None
    head, tail = pair

    dummies = sorted(atom.GetIdx() for atom in mol.GetAtoms()
                     if atom.GetSymbol() == "*")
    bond_type, bond_dir = _link_bond_info(mol, dummies[0], head, dummies[1],
                                          tail)
    n_atoms = mol.GetNumAtoms()

    try:
        if n == 1:
            rw_mol = Chem.RWMol(mol)
        else:
            combined = Chem.Mol(mol)
            for _ in range(n - 1):
                combined = Chem.CombineMols(combined, mol)
            rw_mol = Chem.RWMol(combined)
            for i in range(n - 1):
                tail_i = tail + i * n_atoms
                head_next = head + (i + 1) * n_atoms
                if rw_mol.GetBondBetweenAtoms(tail_i, head_next) is None:
                    rw_mol.AddBond(tail_i, head_next, order=bond_type)
                    if bond_type == Chem.BondType.DOUBLE:
                        new_bond = rw_mol.GetBondBetweenAtoms(
                            tail_i, head_next)
                        if new_bond is not None and bond_dir != Chem.BondDir.NONE:
                            new_bond.SetBondDir(bond_dir)

        removed = sorted(
            [d + i * n_atoms for i in range(n) for d in dummies], reverse=True)
        for idx in removed:
            rw_mol.RemoveAtom(idx)

        oligomer = rw_mol.GetMol()
        Chem.SanitizeMol(oligomer)
        Chem.AssignStereochemistry(oligomer, force=True, cleanIt=True)
    except Exception as e:
        print(f"Error building oligomer (n={n}): {e}")
        return None

    head_idx = _shift_index(head, removed)
    tail_idx = _shift_index(tail + (n - 1) * n_atoms, removed)
    for idx in {head_idx, tail_idx}:
        oligomer.GetAtomWithIdx(idx).SetBoolProp("is_cp", True)

    return oligomer


def choose_oligomer_repeats(
        smiles_or_mol: MolLike,
        depth: int = DEFAULT_DEPTH,
        min_span: Optional[int] = None,
        max_repeats: int = OLIGOMER_MAX_REPEATS) -> Optional[int]:
    """Number of repeat units needed so the span reaches the target (2*depth+1)."""
    span = attachment_topological_distance(smiles_or_mol)
    if span is None:
        return None

    target = min_span if min_span is not None else min_span_for_depth(depth)
    if target <= 0:
        return 1
    if span <= 0 or span >= target:
        return 1

    n = min(max(1, math.ceil((target + 1) / (span + 1))), max_repeats)
    best_n, best_span = 1, span
    while n <= max_repeats:
        oligomer = build_linear_oligomer(smiles_or_mol, n)
        measured = cp_topological_distance(oligomer)
        if measured is None:
            break
        if measured > best_span:
            best_n, best_span = n, measured
        if measured >= target:
            return n
        n += 1

    return best_n


def mapped_cp_smiles(oligomer: Chem.Mol) -> Optional[str]:
    """Oligomer SMILES with the two open ends labeled [1] and [2] for inspection."""
    mol = Chem.Mol(oligomer)
    pair = get_cp_atoms(mol)
    if pair is None:
        return None
    for map_num, idx in zip((1, 2), dict.fromkeys(pair)):
        mol.GetAtomWithIdx(idx).SetAtomMapNum(map_num)
    return Chem.MolToSmiles(mol)


def analyze_smiles(smi: str,
                   depth: int = DEFAULT_DEPTH,
                   max_repeats: int = OLIGOMER_MAX_REPEATS) -> dict:
    """Build the linear oligomer for one repeat unit and record check values."""
    row = {
        "polymer": smi,
        "span_L": None,
        "n_repeats": None,
        "oligomer": None,
        "oligomer_cp_mapped": None,
        "cp_span_measured": None,
        "cp_span_expected": None,
        "span_ok": None,
        "status": "",
    }

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        row["status"] = "invalid_smiles"
        return row
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(dummies) != 2:
        row["status"] = f"expected_2_stars_found_{len(dummies)}"
        return row

    span = attachment_topological_distance(mol)
    row["span_L"] = span

    n = choose_oligomer_repeats(mol, depth=depth, max_repeats=max_repeats)
    row["n_repeats"] = n
    if n is None:
        row["status"] = "no_attachment_points"
        return row

    oligomer = build_linear_oligomer(mol, n)  # linear only, NO ring closure
    if oligomer is None:
        row["status"] = "oligomer_build_failed"
        return row

    row["oligomer"] = Chem.MolToSmiles(oligomer)
    row["oligomer_cp_mapped"] = mapped_cp_smiles(oligomer)
    measured = cp_topological_distance(oligomer)
    row["cp_span_measured"] = measured
    if span is not None and measured is not None:
        row["cp_span_expected"] = n * span + (n - 1)
        row["span_ok"] = measured >= min_span_for_depth(depth)
    row["status"] = "ok"
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Generate linear oligomers (no cyclization) for inspection")
    parser.add_argument("input", nargs="?", default="contrastive_paired.csv",
                        help="input CSV with Polymer_1 / Polymer_2 columns")
    parser.add_argument("-o", "--output", default="oligomer_check.csv",
                        help="output CSV path")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help="D-MPNN depth, span target = 2*depth+1")
    parser.add_argument("--max-repeats", type=int,
                        default=OLIGOMER_MAX_REPEATS)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    smis = []
    for col in ("Polymer_1", "Polymer_2"):
        if col in df.columns:
            smis.extend(df[col].dropna().astype(str))
    unique = list(dict.fromkeys(smis))
    print(f"{len(unique)} unique repeat units from {args.input}")

    rows = [
        analyze_smiles(s, depth=args.depth, max_repeats=args.max_repeats)
        for s in unique
    ]
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    ok = (out["status"] == "ok").sum()
    span_ok = (out["span_ok"] == True).sum()  # noqa: E712
    print(f"written {args.output}: {ok}/{len(out)} built, "
          f"{span_ok} reach span {min_span_for_depth(args.depth)}")
    failures = out[out["status"] != "ok"]["status"].value_counts()
    if len(failures):
        print("failures:", failures.to_dict())


if __name__ == "__main__":
    main()
