# Sugar SMILES finder

A Streamlit app that looks up a sugar by name in [PubChem](https://pubchem.ncbi.nlm.nih.gov/) and shows its SMILES, InChIKey, formula and a 3D structure.

## Features

- **Simple search**: type a name (e.g. `glucose`, `sucrose`) to get the SMILES and the PubChem 3D conformer.
- **Advanced search** (checkbox):
  - Ring puckering: chair (4C1 / 1C4), boat or envelope
  - Anomer: alpha / beta
  - Configuration: D / L

  Conformers are generated with RDKit (ETKDG + MMFF), classified with Cremer-Pople parameters and ranked by energy. If the requested pucker isn't a relaxed minimum, ring-torsion restraints are used to enforce it, and the strain energy is reported. You can download each conformer as a `.mol` file.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```
