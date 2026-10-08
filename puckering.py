"""Sugar ring puckering: Cremer-Pople analysis, IUPAC-style naming and
generation of 3D conformers with a requested ring conformation (RDKit/MMFF)."""

from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

TYPE_NAMES = {"C": "chair", "B": "boat", "S": "skew-boat", "H": "half-chair",
              "E": "envelope", "T": "twist"}
PLANAR_TOL = 0.05  # Å, for classifying ideal ring geometries


# --------------------------------------------------------------------------- #
# Ring perception
# --------------------------------------------------------------------------- #
def sugar_rings(mol):
    """Return sugar rings as (atom indices, labels), ordered O, C(anomeric), ...

    Labels follow carbohydrate numbering: 'O', '1', '2', ... for aldoses and
    'O', '2', '3', ... for ketoses (anomeric carbon carries an exocyclic C).
    """
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        oxy = [a for a in atoms if a.GetSymbol() == "O"]
        if len(oxy) != 1 or any(a.GetSymbol() not in ("C", "O") for a in atoms):
            continue
        ring_set = set(ring)
        o = oxy[0]
        nbrs = [n for n in o.GetNeighbors() if n.GetIdx() in ring_set]

        def exo(atom, symbol):
            return [n for n in atom.GetNeighbors()
                    if n.GetIdx() not in ring_set and n.GetSymbol() == symbol]

        anomeric = max(nbrs, key=lambda a: (len(exo(a, "O")), -len(exo(a, "C"))))
        order = [o.GetIdx(), anomeric.GetIdx()]
        while len(order) < len(ring):
            cur = mol.GetAtomWithIdx(order[-1])
            nxt = [n.GetIdx() for n in cur.GetNeighbors()
                   if n.GetIdx() in ring_set and n.GetIdx() not in order]
            order.append(nxt[0])
        start = 2 if exo(anomeric, "C") else 1
        labels = ["O"] + [str(start + k) for k in range(len(ring) - 1)]
        rings.append((order, labels))
    return rings


# --------------------------------------------------------------------------- #
# Cremer-Pople
# --------------------------------------------------------------------------- #
def _mean_plane(coords):
    n = len(coords)
    c = coords - coords.mean(axis=0)
    ang = 2 * np.pi * np.arange(n) / n
    r1 = (c * np.sin(ang)[:, None]).sum(axis=0)
    r2 = (c * np.cos(ang)[:, None]).sum(axis=0)
    normal = np.cross(r1, r2)
    return c, normal / np.linalg.norm(normal)


def cremer_pople(coords):
    """Return dict with Q, theta (6-rings), phi and z displacements."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    c, normal = _mean_plane(coords)
    z = c @ normal
    j = np.arange(n)
    a = np.sqrt(2 / n) * (z * np.cos(4 * np.pi * j / n)).sum()
    b = -np.sqrt(2 / n) * (z * np.sin(4 * np.pi * j / n)).sum()
    q2 = np.hypot(a, b)
    phi = np.degrees(np.arctan2(b, a)) % 360
    if n == 6:
        q3 = (z * (-1.0) ** j).sum() / np.sqrt(n)
        Q = np.hypot(q2, q3)
        theta = np.degrees(np.arctan2(q2, q3))
        return {"Q": Q, "theta": theta, "phi": phi, "z": z}
    return {"Q": q2, "theta": None, "phi": phi, "z": z}


def ideal_ring(n, theta, phi, Q):
    """Ideal ring coordinates for given Cremer-Pople parameters."""
    j = np.arange(n)
    ang = 2 * np.pi * j / n
    radius = 1.45 / (2 * np.sin(np.pi / n))
    th, ph = np.radians(90.0 if theta is None else theta), np.radians(phi)
    q2 = Q * np.sin(th) if n == 6 else Q
    z = np.sqrt(2 / n) * q2 * np.cos(ph + 4 * np.pi * j / n)
    if n == 6:
        z += Q * np.cos(th) * (-1.0) ** j / np.sqrt(n)
    return np.column_stack([radius * np.cos(ang), -radius * np.sin(ang), z])


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def ring_torsions(coords):
    n = len(coords)
    return [_dihedral(*(coords[(k + i) % n] for i in range(4))) for k in range(n)]


# --------------------------------------------------------------------------- #
# IUPAC-style naming of a ring conformation
# --------------------------------------------------------------------------- #
def _plane_dev(coords, subset):
    pts = coords[list(subset)]
    centred = pts - pts.mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    return np.abs(centred @ normal).max(), normal, pts.mean(axis=0)


def _sup_sub(above, below, kind):
    key = lambda lab: (lab == "O", lab)  # IUPAC writes ring O last, e.g. 3,OB
    sup = ",".join(sorted(above, key=key))
    sub = ",".join(sorted(below, key=key))
    return f"{sup}{kind}{sub}"


def name_conformation(coords, labels, tol=PLANAR_TOL):
    """Name a ring conformation, e.g. '4C1', 'B2,5', 'OS2', '3E', '3T2'."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    _, ref = _mean_plane(coords)  # "above" = side of the Cremer-Pople normal

    def side(atom, normal, centre):
        d = (coords[atom] - centre) @ normal
        return d * np.sign(normal @ ref)

    def priority(exo):  # IUPAC: lowest-numbered ring carbon should be exoplanar
        return sorted(((i - 1) % n) for i in exo)

    # Envelope: n-1 atoms coplanar
    candidates = []
    for subset in combinations(range(n), n - 1):
        dev, normal, centre = _plane_dev(coords, subset)
        if dev < tol:
            flap = (set(range(n)) - set(subset)).pop()
            if abs(side(flap, normal, centre)) > 0.2:
                candidates.append((priority([flap]), flap, normal, centre))
    if candidates:
        _, flap, normal, centre = min(candidates, key=lambda x: x[0])
        if side(flap, normal, centre) > 0:
            return f"{labels[flap]}E"
        return f"E{labels[flap]}"

    # 4-atom (6-ring) or 3-atom (5-ring) reference planes
    size = 4 if n == 6 else 3
    candidates = []
    for subset in combinations(range(n), size):
        if n == 5 and not _contiguous(subset, n):
            continue
        dev, normal, centre = _plane_dev(coords, subset)
        if size == 3 or dev < tol:
            exo = sorted(set(range(n)) - set(subset))
            sides = [side(a, normal, centre) for a in exo]
            if min(abs(s) for s in sides) < 0.2:
                continue
            if n == 5:
                # twist: exoplanar atoms on opposite sides, most symmetric plane
                if sides[0] * sides[1] > 0:
                    continue
                rank = (round(abs(sides[0] + sides[1]), 2),)
            else:
                # IUPAC hierarchy: para pair (C/B) > meta (S) > ortho (H)
                gap = min((exo[1] - exo[0]) % n, (exo[0] - exo[1]) % n)
                rank = (-gap,)
            candidates.append((rank + tuple(priority(exo)), exo, sides, subset))
    if not candidates:
        return "?"
    _, exo, sides, subset = min(candidates, key=lambda x: x[0])
    above = [labels[a] for a, s in zip(exo, sides) if s > 0]
    below = [labels[a] for a, s in zip(exo, sides) if s < 0]
    if n == 5:
        return _sup_sub(above, below, "T")
    a, b = exo
    gap = min((b - a) % n, (a - b) % n)
    if gap == 3:
        kind = "B" if len(above) != 1 else "C"
    elif gap == 1:
        kind = "H"
    else:
        kind = "S"
    return _sup_sub(above, below, kind)


def _contiguous(subset, n):
    s = set(subset)
    return any({k, (k + 1) % n, (k + 2) % n} == s for k in range(n))


def conformation_type(name):
    return next((c for c in name if c in TYPE_NAMES), "?")


# --------------------------------------------------------------------------- #
# Canonical conformations
# --------------------------------------------------------------------------- #
def canonical_conformations(n, labels):
    """List of dicts {name, type, theta, phi, Q} of canonical conformers."""
    out = {}
    if n == 6:
        grid = [(0, 0), (180, 0)]
        grid += [(90, p) for p in range(0, 360, 30)]
        grid += [(t, p) for t in (50.8, 54.7, 125.3, 129.2) for p in range(0, 360, 30)]
        q = {0: 0.57, 180: 0.57}
    else:
        grid = [(None, p) for p in range(0, 360, 18)]
        q = {}
    for theta, phi in grid:
        Q = q.get(theta, 0.65 if n == 6 else 0.40)
        name = name_conformation(ideal_ring(n, theta, phi, Q), labels)
        kind = conformation_type(name)
        expected = {0: "C", 180: "C", 90: "BS", 50.8: "H", 129.2: "H",
                    54.7: "E", 125.3: "E", None: "ET"}[theta]
        if kind in expected and name not in out:
            out[name] = {"name": name, "type": kind, "theta": theta, "phi": phi, "Q": Q}
    return list(out.values())


def classify(coords, labels):
    """Closest canonical conformation for real (non-ideal) coordinates."""
    n = len(coords)
    cp = cremer_pople(coords)
    best, best_d = None, np.inf
    for c in canonical_conformations(n, labels):
        if n == 6:
            t1, t2 = np.radians(cp["theta"]), np.radians(c["theta"])
            p1, p2 = np.radians(cp["phi"]), np.radians(c["phi"])
            d = np.arccos(np.clip(np.cos(t1) * np.cos(t2)
                                  + np.sin(t1) * np.sin(t2) * np.cos(p1 - p2), -1, 1))
        else:
            d = abs((cp["phi"] - c["phi"] + 180) % 360 - 180)
        if d < best_d:
            best, best_d = c, d
    return best["name"], cp


# --------------------------------------------------------------------------- #
# Conformer generation
# --------------------------------------------------------------------------- #
def _ring_coords(conf, ring):
    return np.array([list(conf.GetAtomPosition(i)) for i in ring])



# --------------------------------------------------------------------------- #
# Conformer generation (ensemble search -> filter -> rank by MMFF energy)
# --------------------------------------------------------------------------- #
CHAIR_THETA = {"4C1": 0.0, "1C4": 180.0}
ALLOWED = {"C": (6,), "B": (6,), "E": (5, 6)}


def _ring_coords(conf, ring):
    return np.array([list(conf.GetAtomPosition(i)) for i in ring])


def _analyse(mol, conf_id, rings):
    conf = mol.GetConformer(conf_id)
    out = []
    for ring, labels in rings:
        name, cp = classify(_ring_coords(conf, ring), labels)
        out.append({"name": name, "type": conformation_type(name), "cp": cp,
                    "size": len(ring)})
    return out


def _matches(ring_info, ptype, chair):
    relevant = [r for r in ring_info if r["size"] in ALLOWED[ptype]]
    for r in relevant:
        if r["type"] != ptype:
            return False
        if ptype == "C" and chair:
            want_low = CHAIR_THETA[chair] < 90
            if (r["cp"]["theta"] < 90) != want_low:
                return False
    return bool(relevant)


def _single(mol, conf_id):
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    m.AddConformer(Chem.Conformer(mol.GetConformer(conf_id)), assignId=True)
    return m


def _constrained(mol, props, rings, start_ids, ptype, chair, window, k):
    """Force the requested pucker with flat-bottom MMFF ring-torsion restraints."""
    first = next((r for r in rings if len(r[0]) in ALLOWED[ptype]), None)
    n = len(first[0])
    variants = [c for c in canonical_conformations(n, first[1]) if c["type"] == ptype]
    if ptype == "C" and chair:
        variants = [c for c in variants if c["theta"] == CHAIR_THETA[chair]]
    found = []
    for var in variants:
        best = None
        for cid in start_ids:
            work = _single(mol, cid)
            ff = AllChem.MMFFGetMoleculeForceField(work, props)
            for ring, _ in rings:
                if len(ring) != n:
                    continue  # rings of other size are left free
                ideal = ring_torsions(ideal_ring(n, var["theta"], var["phi"], var["Q"]))
                for j, tor in enumerate(ideal):
                    quad = [ring[(j + i) % n] for i in range(4)]
                    ff.MMFFAddTorsionConstraint(*quad, False, tor - window, tor + window, k)
            ff.Minimize(maxIts=10000)
            energy = AllChem.MMFFGetMoleculeForceField(work, props).CalcEnergy()
            if best is None or energy < best[0]:
                best = (energy, work)
        found.append(best)
    return found


def generate_conformers(smiles, ptype=None, chair=None, num_conformers=60,
                        max_keep=5, rmsd_threshold=0.5, seed=42,
                        window=12.0, k=500.0):
    """Generate sugar conformers, optionally restricted to a ring pucker.

    ptype: None (any), 'C' (chair), 'B' (boat) or 'E' (envelope).
    chair: '4C1' or '1C4' (only used with ptype='C'; for ketoses these map
           to theta=0 / theta=180, i.e. 5C2 / 2C5).

    Returns (results, info). Each result has 'mol' (single conformer with H),
    'energy', 'dE' (kcal/mol above the unrestrained global minimum found),
    'rings' (per-ring name + Cremer-Pople) and 'source' ('ensemble' if it is a
    relaxed MMFF minimum, 'constrained' if the pucker had to be enforced).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES.")
    mol = Chem.AddHs(mol)
    rings = sugar_rings(mol)
    if not rings:
        raise ValueError("No sugar ring (oxane/oxolane) found in this molecule.")
    if ptype and not any(len(r[0]) in ALLOWED[ptype] for r in rings):
        sizes = sorted({len(r[0]) for r in rings})
        raise ValueError(f"A {TYPE_NAMES[ptype]} is not possible for "
                         f"{'/'.join(map(str, sizes))}-membered sugar rings.")

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))
    if not cids:
        raise ValueError("RDKit could not embed this molecule in 3D.")
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        raise ValueError("MMFF parameters are not available for this molecule.")
    opt = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=5000)
    energies = {cid: e for cid, (_, e) in zip(cids, opt)}
    e_min = min(energies.values())

    ensemble = [(energies[c], _single(mol, c), _analyse(mol, c, rings)) for c in cids]
    ensemble.sort(key=lambda x: x[0])
    counts = {}
    for _, _, ri in ensemble:
        key = " / ".join(r["name"] for r in ri)
        counts[key] = counts.get(key, 0) + 1

    if ptype:
        hits = [(e, m, "ensemble") for e, m, ri in ensemble if _matches(ri, ptype, chair)]
        if not hits:
            starts = [c for c in sorted(cids, key=energies.get)][:5]
            hits = [(e, m, "constrained") for e, m in
                    _constrained(mol, props, rings, starts, ptype, chair, window, k)]
    else:
        hits = [(e, m, "ensemble") for e, m, _ in ensemble]
    hits.sort(key=lambda x: x[0])

    results, kept_heavy = [], []
    for energy, m, source in hits:
        heavy = Chem.RemoveHs(m)
        if any(rdMolAlign.GetBestRMS(heavy, h) < rmsd_threshold for h in kept_heavy):
            continue
        kept_heavy.append(heavy)
        results.append({"mol": m, "energy": energy, "dE": energy - e_min,
                        "rings": _analyse(m, 0, rings), "source": source})
        if len(results) >= max_keep:
            break
    info = {"n_ensemble": len(cids), "e_min": e_min, "ensemble_counts": counts,
            "ring_labels": [labels for _, labels in rings]}
    return results, info


def ensure_3d(molblock_or_smiles, is_smiles=False, seed=42):
    """Return a single MMFF-optimised conformer (used when PubChem has no 3D)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(molblock_or_smiles)) if is_smiles \
        else Chem.MolFromMolBlock(molblock_or_smiles, removeHs=False)
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def substituent_orientation(mol):
    """Axial/equatorial orientation of exocyclic heavy atoms per sugar ring."""
    conf = mol.GetConformer()
    out = []
    for ring, labels in sugar_rings(mol):
        _, normal = _mean_plane(_ring_coords(conf, ring))
        ring_set = set(ring)
        info = {}
        for idx, lab in zip(ring, labels):
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetIdx() in ring_set or nb.GetAtomicNum() == 1:
                    continue
                v = np.array(conf.GetAtomPosition(nb.GetIdx())) - np.array(conf.GetAtomPosition(idx))
                axial = abs(v @ normal) / np.linalg.norm(v) > 0.7
                info.setdefault(lab, []).append("ax" if axial else "eq")
        out.append({k: "/".join(v) for k, v in info.items()})
    return out
