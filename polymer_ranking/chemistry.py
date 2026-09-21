"""Chemistry preprocessing: cyclization, extra feature extraction, DataFrame preprocessing."""

import logging
import math
from collections import deque
from typing import FrozenSet, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem

from .config import (
    EXTRA_COLS,
    TASK_NAMES,
    DEFAULT_DEPTH,
    OLIGOMER_MAX_REPEATS,
    CENSOR_LOG_MARGIN,
    min_span_for_depth,
)

logger = logging.getLogger(__name__)

MolLike = Union[str, Chem.Mol]


def _as_mol(smiles_or_mol: MolLike) -> Optional[Chem.Mol]:
    """Accept a SMILES string or an RDKit Mol and always return a Mol (None if invalid)."""
    if isinstance(smiles_or_mol, Chem.Mol):
        return smiles_or_mol
    if not isinstance(smiles_or_mol, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles_or_mol)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Error parsing smiles {smiles_or_mol}: {e}")
        return None


def cyclize_polymer_with_cp_marking(
        smiles: str,
        depth: int = DEFAULT_DEPTH,
        min_span: Optional[int] = None,
        max_repeats: int = OLIGOMER_MAX_REPEATS,
        auto_oligomer: bool = True) -> Tuple[Optional[str], Optional[Chem.Mol]]:
    """
    Cyclize a polymer repeat unit (with *) and mark connection points (CP).

    Cyclization collapses the backbone span L between the two attachment atoms
    to a single bond. When ``L < 2 * depth + 1`` that shortcut lies inside the
    receptive field of a ``depth``-layer D-MPNN, so the unit is first expanded
    head-to-tail into an oligomer of ``n`` repeat units (``n <= max_repeats``)
    and only the two ends of the oligomer are joined.

    Parameters
    ----------
    smiles : str
        Repeat unit SMILES with exactly two ``*`` attachment points.
    depth : int
        Number of D-MPNN message passing steps; sets the required span
        ``2 * depth + 1`` unless ``min_span`` is given explicitly.
    min_span : Optional[int]
        Explicit span requirement, overrides the ``depth`` based value.
    max_repeats : int
        Maximum number of repeat units chained for a too short unit.
    auto_oligomer : bool
        Set to False to always cyclize the single repeat unit (legacy behavior).

    Returns
    -------
    Tuple[smiles, mol]
        - smiles: Cyclized SMILES string
        - mol: Cyclized RDKit Mol object with connection point atoms marked with
          is_cp property and the number of repeat units stored in n_repeat.
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

    pair = get_attachment_atoms(mol)
    if pair is None:
        return None, None
    n1_idx, n2_idx = pair

    try:
        bond_type, bond_dir = _link_bond_info(mol, dummy_indices[0], n1_idx,
                                              dummy_indices[1], n2_idx)

        repeats = 1
        if auto_oligomer:
            chosen = choose_oligomer_repeats(
                mol,
                depth=depth,
                min_span=min_span,
                max_repeats=max_repeats,
            )
            if chosen is not None and chosen > 1:
                repeats = chosen

        if repeats > 1:
            cyclic_mol = _cyclize_oligomer(mol, repeats, bond_type, bond_dir)
            if cyclic_mol is not None:
                cyclic_mol.SetIntProp("n_repeat", repeats)
                return Chem.MolToSmiles(cyclic_mol), cyclic_mol
            logger.warning(
                f"Oligomer cyclization (n={repeats}) failed for {smiles}; "
                f"falling back to the single repeat unit")

        rw_mol = Chem.RWMol(mol)

        # If n1 == n2 (shared neighbor) or the two atoms are already bonded,
        # just remove the dummies — no new bond is needed
        _close_ring(rw_mol, n1_idx, n2_idx, bond_type, bond_dir)

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
        cyclic_mol.SetIntProp("n_repeat", 1)

        return Chem.MolToSmiles(cyclic_mol), cyclic_mol

    except Exception as e:
        logger.error(f"Error processing smiles {smiles}: {e}")
        return None, None


def get_attachment_atoms(smiles_or_mol: MolLike) -> Optional[Tuple[int, int]]:
    """
    Locate the two attachment atoms of a polymer repeat unit.

    An attachment atom is the (normally carbon) heavy atom bonded to one of the
    two ``*`` dummy atoms, i.e. the atom that will carry the bond formed by
    cyclization.

    Parameters
    ----------
    smiles_or_mol : str or Chem.Mol
        Repeat unit with exactly two ``*`` attachment points.

    Returns
    -------
    Optional[Tuple[int, int]]
        Atom indices of the two attachment atoms in the input numbering,
        or None if the input is invalid or does not hold exactly two
        attachment points (e.g. already cyclized, 0/1/3+ dummies,
        dangling ``*``).
    """
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


def attachment_topological_distance(smiles_or_mol: MolLike) -> Optional[int]:
    """
    Topological distance L between the two attachment atoms of a repeat unit.

    L is the number of bonds on the shortest path connecting the two
    attachment atoms in the **uncyclized** unit graph (the ``*`` dummy atoms
    themselves are excluded from the graph). It is the span the unit covers
    along the backbone; cyclization collapses that span to 1.

    This is the quantity that decides whether cyclization is safe for a
    D-MPNN of ``depth`` message passing steps: information travels at most
    ``depth`` bonds per step, so the unit should satisfy ``L >= 2 * depth + 1``
    (13 for the default ``depth=6``). Smaller units must first be expanded
    into an oligomer before cyclization.

    Parameters
    ----------
    smiles_or_mol : str or Chem.Mol
        Repeat unit with exactly two ``*`` attachment points.

    Returns
    -------
    Optional[int]
        L in bonds. ``0`` when both attachment points sit on the same atom,
        ``None`` when the input is invalid or has no two attachment points.
    """
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


def _shortest_bond_distance(
        mol: Chem.Mol,
        start: int,
        end: int,
        blocked: FrozenSet[int] = frozenset()) -> Optional[int]:
    """
    Number of bonds on the shortest path between two atoms (BFS).

    Atoms listed in ``blocked`` (e.g. the ``*`` dummies) are excluded, so the
    reported span never uses a shortcut through an attachment point.
    Returns None when the two atoms are not connected.
    """
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


def get_cp_atoms(mol: Optional[Chem.Mol]) -> Optional[Tuple[int, int]]:
    """
    Return the two atoms marked with ``is_cp`` (connection points).

    Works on linear oligomers produced by :func:`build_linear_oligomer` as well
    as on cyclized molecules. When both attachment points share a single atom
    that index is returned twice.
    """
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
    """
    Bond type / direction used for the bond that replaces a ``*`` attachment.

    Follows the same convention as cyclization: the dummy bonds are single in
    virtually every polymer SMILES, so the resulting link is a single bond
    unless one of the dummy bonds is not.
    """
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


def _close_ring(rw_mol: Chem.RWMol, i: int, j: int, bond_type: Chem.BondType,
                bond_dir: Chem.BondDir) -> bool:
    """Join two attachment atoms, no-op when they coincide or are already bonded."""
    if i == j:
        return True
    if rw_mol.GetBondBetweenAtoms(i, j) is not None:
        return True
    rw_mol.AddBond(i, j, order=bond_type)
    new_bond = rw_mol.GetBondBetweenAtoms(i, j)
    if new_bond is None:
        return False
    if bond_dir != Chem.BondDir.NONE:
        new_bond.SetBondDir(bond_dir)
    return True


def _shift_index(idx: int, removed) -> int:
    """Map an atom index to its value after the indices in ``removed`` are gone."""
    return idx - sum(1 for r in removed if r < idx)


def build_linear_oligomer(smiles_or_mol: MolLike,
                          n: int = 1) -> Optional[Chem.Mol]:
    """
    Chain ``n`` copies of a repeat unit head-to-tail and strip the ``*`` dummies.

    The result is still open: the head of the first copy and the tail of the
    last copy are marked with ``is_cp`` so that they can be joined afterwards
    (see :func:`cyclize_polymer_with_cp_marking`). ``n == 1`` simply returns the
    de-dummied repeat unit.

    Returns
    -------
    Optional[Chem.Mol]
        Sanitized linear oligomer with the two open ends marked, or None.
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
        logger.error(f"Error building oligomer (n={n}): {e}")
        return None

    head_idx = _shift_index(head, removed)
    tail_idx = _shift_index(tail + (n - 1) * n_atoms, removed)
    for idx in {head_idx, tail_idx}:
        oligomer.GetAtomWithIdx(idx).SetBoolProp("is_cp", True)

    return oligomer


def _cyclize_oligomer(unit_mol: Chem.Mol, n: int, bond_type: Chem.BondType,
                      bond_dir: Chem.BondDir) -> Optional[Chem.Mol]:
    """Build the n-mer of ``unit_mol`` and close it into a ring. None on failure."""
    oligomer = build_linear_oligomer(unit_mol, n)
    if oligomer is None:
        return None
    pair = get_cp_atoms(oligomer)
    if pair is None:
        return None
    try:
        rw_mol = Chem.RWMol(oligomer)
        if not _close_ring(rw_mol, pair[0], pair[1], bond_type, bond_dir):
            return None
        cyclic_mol = rw_mol.GetMol()
        Chem.SanitizeMol(cyclic_mol)
        Chem.AssignStereochemistry(cyclic_mol, force=True, cleanIt=True)
        return cyclic_mol
    except Exception as e:
        logger.error(f"Error cyclizing oligomer (n={n}): {e}")
        return None


def choose_oligomer_repeats(
        smiles_or_mol: MolLike,
        depth: int = DEFAULT_DEPTH,
        min_span: Optional[int] = None,
        max_repeats: int = OLIGOMER_MAX_REPEATS) -> Optional[int]:
    """
    Number of repeat units needed before cyclization is safe.

    The span of an n-mer is ``n * L + (n - 1)`` bonds (L = span of one unit),
    so ``n = ceil((target + 1) / (L + 1))`` is the theoretical answer. The
    value is then verified on the actually built oligomer and increased if the
    measurement disagrees (e.g. when the unit graph changes after the dummies
    are stripped).

    Returns
    -------
    Optional[int]
        Repeat count in ``[1, max_repeats]``; ``1`` when the unit is already
        long enough, ``None`` when the input has no two attachment points.
        When even ``max_repeats`` copies cannot reach the target, the repeat
        count with the largest measured span is returned.
    """
    span = attachment_topological_distance(smiles_or_mol)
    if span is None:
        return None

    target = min_span if min_span is not None else min_span_for_depth(depth)
    if target <= 0:
        return 1
    # L == 0 means both attachment points sit on the same atom; chaining such a
    # unit is meaningless, keep the single unit.
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

    if best_n > 1:
        logger.debug(
            f"Span target {target} not reached within {max_repeats} repeats "
            f"(best span {best_span} at n={best_n})")
    return best_n


def cyclize_df(
        df: pd.DataFrame,
        depth: int = DEFAULT_DEPTH,
        min_span: Optional[int] = None,
        max_repeats: int = OLIGOMER_MAX_REPEATS,
        auto_oligomer: bool = True) -> pd.DataFrame:
    """Perform cyclization on DataFrame, add cyc_1/2, mol_1/2 columns."""
    df = df.copy()
    n_expanded = 0
    for s in ["1", "2"]:
        results = df[f"Polymer_{s}"].apply(
            lambda smi: cyclize_polymer_with_cp_marking(
                smi,
                depth=depth,
                min_span=min_span,
                max_repeats=max_repeats,
                auto_oligomer=auto_oligomer))
        df[f"cyc_{s}"] = results.apply(lambda x: x[0])
        df[f"mol_{s}"] = results.apply(lambda x: x[1])
        n_expanded += int(
            sum(1 for m in df[f"mol_{s}"]
                if m is not None and m.HasProp("n_repeat")
                and m.GetIntProp("n_repeat") > 1))

    if auto_oligomer:
        logger.info(
            f"Cyclization: {n_expanded} short units expanded into oligomers "
            f"(span target {min_span if min_span is not None else min_span_for_depth(depth)}, "
            f"max_repeats={max_repeats})")
    return df


def extra_feat(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """Extract extra feature columns, fill NaN with 0."""
    cols = [c.format(s=suffix) for c in EXTRA_COLS]
    return df[cols].fillna(0.0).values.astype(np.float32)


def load_and_preprocess(
        csv_path: str,
        depth: int = DEFAULT_DEPTH,
        min_span: Optional[int] = None,
        max_repeats: int = OLIGOMER_MAX_REPEATS,
        auto_oligomer: bool = True) -> pd.DataFrame:
    """
    Read CSV and perform:
      - Polymer cyclization (with oligomer expansion for too short units)
      - Take log10 of mobility columns
      - Drop invalid rows

    Censored labels
    ---------------
    A mobility of ``0`` is **left-censored** (below the detection limit), not a
    missing measurement. Such a value still carries ordering information — it is
    lower than every measured mobility — but it carries no numeric information.
    Therefore

    * ``ok_{task}_{side}`` (bool) marks a measured (``> 0``) mobility;
    * ``log_{task}_{side}`` of a censored entry is filled with a sentinel placed
      ``CENSOR_LOG_MARGIN`` decades *below* the smallest measured value of that
      task. The sentinel only makes ``sign(y1 - y2)`` come out right for
      measured-vs-censored pairs; the regression term never uses it because the
      loss masks those pairs with ``ok``.

    Rows whose two mobilities are both censored are kept: they are still usable
    as (unsupervised) structure pairs only if at least one task is decidable,
    otherwise they contribute nothing to either term.
    """
    df = pd.read_csv(csv_path)
    df = cyclize_df(
        df,
        depth=depth,
        min_span=min_span,
        max_repeats=max_repeats,
        auto_oligomer=auto_oligomer)

    target_raw = [f"{t}_{s}" for t in TASK_NAMES for s in ("1", "2")]
    df = df.dropna(subset=["cyc_1", "cyc_2"] +
                   target_raw).reset_index(drop=True)
    df = df[df["mol_1"].notna() & df["mol_2"].notna()].reset_index(drop=True)

    for t in TASK_NAMES:
        raw = pd.concat([df[f"{t}_1"], df[f"{t}_2"]])
        measured = raw[raw > 0]
        if len(measured):
            floor = float(np.log10(measured.min())) - CENSOR_LOG_MARGIN
        else:  # pragma: no cover - degenerate dataset without a single label
            floor = -6.0 - CENSOR_LOG_MARGIN
        for s in ("1", "2"):
            col = f"{t}_{s}"
            ok = (df[col] > 0).values
            vals = np.array(np.log10(df[col].clip(lower=1e-30)), dtype=np.float64)
            vals[~ok] = floor
            df[f"ok_{col}"] = ok
            df[f"log_{col}"] = vals.astype(np.float64)

        ok1, ok2 = df[f"ok_{t}_1"], df[f"ok_{t}_2"]
        n_both = int((ok1 & ok2).sum())
        n_one = int((ok1 ^ ok2).sum())
        n_none = int((~ok1 & ~ok2).sum())
        logger.info(
            f"{t}: {n_both}/{len(df)} pairs have both mobilities measured "
            f"(regression target known); {n_one} have exactly one censored side "
            f"(ranking only); {n_none} are censored on both sides (no ordering "
            f"information). censored log floor = {floor:.2f}"
        )

    logger.info(f"Preprocessed dataset: {len(df)} valid pairs")
    return df
