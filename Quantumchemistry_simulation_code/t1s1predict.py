#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch electronic-state energies with MLatom/AIQM1
Usage: python T1S1.py
- Expects files named dsgdb9nsd_000001.xyz ... dsgdb9nsd_133885.xyz in INPUT_DIR.
- Writes per-molecule outputs to OUTPUT_DIR as "<stem>_energies.txt" with:
  Electronic state energies in singlet: [E1 E2 E3]
  Electronic state energies in triplet : [E1 E2 E3]

terminal: export mndobin=/aitomistic/hub/home/.../mndo/BIN/mndo2020_2020-09-10_ifort-13.1.3.192_mkl-11.1.4.211
python T1S1newtestbank/t1s1predict.py
"""

import os
import sys
import numpy as np

try:
    import mlatom as ml
except Exception as e:
    print("ERROR: Could not import mlatom. Make sure MLatom is installed in this environment.")
    print("Original error:", repr(e))
    sys.exit(1)

# ---------------- User-adjustable paths ----------------
INPUT_DIR = "/aitomistic/hub/home/.../T1S1newtestbank/input/converted_xyz"                # Folder containing the .xyz files
OUTPUT_DIR = "/aitomistic/hub/home/.../T1S1newtestbank/output"  # Folder to write the *_energies.txt files
# -------------------------------------------------------

START_INDEX = 1
END_INDEX = 1000
NSTATES = 3  # number of electronic states to compute in each multiplicity

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reuse a single AIQM1 model for all predictions
aiqm1 = ml.models.methods(method='AIQM1')

def read_molecule(xyz_path: str, multiplicity: int):
    """Create & return an MLatom molecule from an XYZ, with given multiplicity."""
    mol = ml.data.molecule()
    mol.read_from_xyz_file(xyz_path)
    mol.charge = 0
    mol.multiplicity = multiplicity  # 1 = singlet, 3 = triplet
    return mol

def compute_state_energies(mol) -> np.ndarray:
    """Run AIQM1 and return the state energies as a NumPy array (shape: (NSTATES,))."""
    aiqm1.predict(
        molecule=mol,
        nstates=3,
        current_state=0,
        calculate_energy=True,
        calculate_nacv=False,
        read_density_matrix=False,
    )
    energies = getattr(mol, "state_energies", None)
    if energies is None:
        raise RuntimeError("No 'state_energies' found on molecule after prediction.")
    return np.array(energies, dtype=float)

def main():
    missing = []
    failed = []

    # Loop through numeric filenames 1.xyz, 2.xyz, etc.
    for i in range(START_INDEX, END_INDEX + 1):
        xyz_path = os.path.join(INPUT_DIR, f"{i}.xyz")
        stem = str(i)

        if not os.path.exists(xyz_path):
            missing.append(stem)
            continue

        try:
            # Singlet energies
            mol_s = read_molecule(xyz_path, multiplicity=1)
            e_s = compute_state_energies(mol_s)

            # Triplet energies
            mol_t = read_molecule(xyz_path, multiplicity=3)
            e_t = compute_state_energies(mol_t)

            # Format arrays exactly like the example (NumPy prints without commas by default)
            singlet_str = np.array2string(e_s, separator=' ', max_line_width=10**9)
            triplet_str = np.array2string(e_t, separator=' ', max_line_width=10**9)

            out_path = os.path.join(OUTPUT_DIR, stem + "_energies.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Electronic state energies in singlet: {singlet_str}\n")
                f.write(f"Electronic state energies in triplet : {triplet_str}\n")
            
            print(i)

        except Exception as ex:
            failed.append((stem, str(ex)))



    # Logs for missing/failed cases
    if missing:
        with open(os.path.join(OUTPUT_DIR, "_missing_files.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
    if failed:
        with open(os.path.join(OUTPUT_DIR, "_failed_files.log"), "w", encoding="utf-8") as f:
            for stem, err in failed:
                f.write(f"{stem}\t{err}\n")

    print("Done.")
    if missing:
        print(f"Missing files: {len(missing)} (see {os.path.join(OUTPUT_DIR, '_missing_files.log')})")
    if failed:
        print(f"Failed molecules: {len(failed)} (see {os.path.join(OUTPUT_DIR, '_failed_files.log')})")

if __name__ == "__main__":
    main()
