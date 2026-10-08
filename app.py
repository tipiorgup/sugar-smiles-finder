import re
from urllib.parse import quote

import py3Dmol
import requests
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem

import puckering as pk

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPERTIES = "SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,IUPACName,InChIKey"
PREFIX_RE = re.compile(r"^\s*(?:(alpha|beta|α|β)\s*-\s*)?(?:([DL])\s*-\s*)?", re.I)
PUCKER_CODES = {"Chair": "C", "Boat": "B", "Envelope": "E"}


# --------------------------------------------------------------------------- #
# PubChem
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def search_pubchem(name: str) -> list[dict]:
    resp = requests.get(f"{PUBCHEM}/compound/name/{quote(name, safe='')}"
                        f"/property/{PROPERTIES}/JSON", timeout=20)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json()["PropertyTable"]["Properties"]


@st.cache_data(show_spinner=False)
def pubchem_3d(cid: int) -> str | None:
    resp = requests.get(f"{PUBCHEM}/compound/cid/{cid}/record/SDF",
                        params={"record_type": "3d"}, timeout=20)
    return resp.text if resp.ok else None


def build_query(name: str, anomer: str, config: str) -> str:
    """e.g. ('D-glucose', 'Alpha', 'Any') -> 'alpha-D-glucose'."""
    m = PREFIX_RE.match(name)
    typed_anomer, typed_config = m.group(1), m.group(2)
    base = name[m.end():].strip() or name.strip()
    if anomer == "Any" and typed_anomer:
        anomer = {"α": "alpha", "β": "beta"}.get(typed_anomer, typed_anomer)
    if config == "Any" and typed_config:
        config = typed_config.upper()
    parts = [p for p in (anomer.lower() if anomer != "Any" else "",
                         config if config != "Any" else "") if p]
    return "-".join(parts + [base])


# --------------------------------------------------------------------------- #
# Conformers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def conformers(smiles: str, ptype: str | None, chair: str | None):
    results, info = pk.generate_conformers(smiles, ptype, chair)
    rows = []
    for r in results:
        rows.append({
            "molblock": Chem.MolToMolBlock(r["mol"]),
            "dE": r["dE"],
            "source": r["source"],
            "rings": [{"name": x["name"], "size": x["size"], "Q": x["cp"]["Q"],
                       "theta": x["cp"]["theta"], "phi": x["cp"]["phi"]} for x in r["rings"]],
            "orientation": pk.substituent_orientation(r["mol"]),
        })
    return rows, {"n_ensemble": info["n_ensemble"]}


@st.cache_data(show_spinner=False)
def rdkit_3d(smiles: str) -> str:
    return Chem.MolToMolBlock(pk.ensure_3d(smiles, is_smiles=True))


def show_3d(molblock: str, height: int = 420):
    view = py3Dmol.view(width=680, height=height)
    view.addModel(molblock, "mol")
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.22}})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=height + 10)


def show_properties(r: dict):
    cid = r["CID"]
    st.subheader(f"CID {cid}")
    st.markdown(f"[Open in PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
    st.write("**SMILES (isomeric):**")
    st.code(r.get("SMILES", "N/A"), language=None)
    st.write("**SMILES (connectivity, no stereo):**")
    st.code(r.get("ConnectivitySMILES", "N/A"), language=None)
    st.write(f"**Formula:** {r.get('MolecularFormula', 'N/A')}  |  "
             f"**MW:** {r.get('MolecularWeight', 'N/A')}")
    st.write(f"**IUPAC name:** {r.get('IUPACName', 'N/A')}")
    st.write(f"**InChIKey:** {r.get('InChIKey', 'N/A')}")


def show_simple_3d(r: dict):
    st.markdown("#### 3D structure (PubChem conformer)")
    sdf = pubchem_3d(r["CID"])
    if sdf:
        show_3d(sdf)
    else:
        st.caption("PubChem has no 3D conformer for this compound; showing an RDKit/MMFF structure.")
        show_3d(rdkit_3d(r["SMILES"]))


def show_advanced_3d(r: dict, ptype: str | None, chair: str | None):
    cid = r["CID"]
    st.markdown("#### 3D conformers (RDKit ETKDG + MMFF, ranked by energy)")
    with st.spinner("Generating and ranking conformers..."):
        try:
            rows, info = conformers(r["SMILES"], ptype, chair)
        except ValueError as e:
            st.error(str(e))
            return
    if not rows:
        st.warning("No conformer found.")
        return

    table = [{
        "#": i + 1,
        "Ring conformation": " / ".join(x["name"] for x in row["rings"]),
        "ΔE (kcal/mol)": round(row["dE"], 2),
        "Source": "relaxed minimum" if row["source"] == "ensemble" else "restrained (forced pucker)",
    } for i, row in enumerate(rows)]
    st.dataframe(table, hide_index=True, use_container_width=True)
    pick = st.selectbox(
        "Conformer to display", range(len(rows)), key=f"pick{cid}",
        format_func=lambda i: f"#{i + 1}  {table[i]['Ring conformation']}  "
                              f"(ΔE = {table[i]['ΔE (kcal/mol)']} kcal/mol)")
    row = rows[pick]
    show_3d(row["molblock"])

    for ring, orient in zip(row["rings"], row["orientation"]):
        theta = f"θ = {ring['theta']:.1f}°, " if ring["theta"] is not None else ""
        st.write(f"**{ring['size']}-membered ring: {ring['name']}** — Cremer–Pople "
                 f"Q = {ring['Q']:.3f} Å, {theta}φ = {ring['phi']:.1f}°")
        if ring["size"] == 6:
            st.caption("Substituents: " + ", ".join(f"C{k}: {v}" for k, v in orient.items()))
    if row["source"] != "ensemble":
        st.info("The requested pucker is not a relaxed MMFF minimum for this sugar, so it was "
                "enforced with flat-bottom ring-torsion restraints. ΔE is the strain relative "
                "to the lowest-energy conformer found.")
    st.caption(f"{info['n_ensemble']} conformers sampled; ΔE is relative to the lowest-energy "
               "conformer found. SMILES does not encode ring puckering — the conformation "
               "only exists in the 3D file.")
    fname = table[pick]["Ring conformation"].replace(" / ", "_").replace(",", "-")
    st.download_button("Download 3D structure (.mol)", row["molblock"],
                       file_name=f"CID{cid}_{fname}.mol", key=f"dl{cid}")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Sugar SMILES finder", page_icon="🍬")
st.title("🍬 Sugar SMILES finder")
st.caption("Type a sugar name to look it up in PubChem and get its SMILES and 3D structure.")

name = st.text_input("Sugar name", placeholder="e.g. glucose, sucrose, D-fructose")
advanced = st.checkbox("Advanced search")

ptype = chair = None
anomer = config = "Any"
if advanced:
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            pucker = st.radio("Ring puckering", ["Any", "Chair", "Boat", "Envelope"])
            if pucker == "Chair":
                chair = st.radio("Chair conformation", ["4C1", "1C4"], horizontal=True,
                                 format_func=lambda c: {"4C1": "⁴C₁", "1C4": "¹C₄"}[c])
        with col2:
            anomer = st.radio("Anomer", ["Any", "Alpha", "Beta"], horizontal=True,
                              format_func=lambda a: {"Any": "Any", "Alpha": "α", "Beta": "β"}[a])
            config = st.radio("Configuration", ["Any", "D", "L"], horizontal=True)
        ptype = PUCKER_CODES.get(pucker)

if st.button("Search", type="primary"):
    if not name.strip():
        st.info("Please enter a name.")
        st.session_state.pop("search", None)
    else:
        query = build_query(name.strip(), anomer, config) if advanced else name.strip()
        st.session_state["search"] = {"query": query, "advanced": advanced,
                                      "ptype": ptype, "chair": chair}

search = st.session_state.get("search")
if search:
    query = search["query"]
    with st.spinner(f"Searching PubChem for '{query}'..."):
        try:
            results = search_pubchem(query)
        except requests.RequestException as e:
            st.error(f"PubChem request failed: {e}")
            st.stop()

    if not results:
        st.warning(f"No compound found in PubChem for **{query}**. Try another name or relax "
                   "the anomer / D-L options (e.g. sucrose doesn't take an α/β prefix).")
        st.stop()
    if search["advanced"]:
        st.success(f"PubChem query: **{query}**")

    for r in results:
        show_properties(r)
        if search["advanced"]:
            show_advanced_3d(r, search["ptype"], search["chair"])
        else:
            show_simple_3d(r)
        st.divider()
