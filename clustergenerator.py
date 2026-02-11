# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:29:47 2025

@author: Windows
"""

import os
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io import read, write

# ========== PARAMETERS ==========
poscar_path = 'POSCAR'       # Input POSCAR file, must contain exactly 1 Ag atom
bond_distance = 2.8          # Distance from existing Ag atom to place a new Ag atom
min_dist_to_any_atom = 2.4   # Minimum allowed distance to any atom to avoid overlap
n_points = 32                # Number of directions to sample on a unit sphere
max_ag_atoms = 5            # Maximum number of Ag atoms to place
# ================================

def fibonacci_sphere(samples=32):
    """
    Generate `samples` points evenly distributed on a unit sphere surface
    using the Fibonacci lattice algorithm.
    """
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append((x, y, z))
    return np.array(points)

def is_valid_position(new_pos, existing_positions, min_dist):
    """
    Check whether the new position is at least `min_dist` away from all existing atoms.
    """
    dists = np.linalg.norm(existing_positions - new_pos, axis=1)
    return np.all(dists >= min_dist)

def generate_next_ag_structures(structures_with_path, ag_count):
    """
    For each existing structure, generate new structures by adding one more Ag atom
    at sampled directions around each current Ag atom.
    
    Parameters:
        structures_with_path (list): List of (Atoms object, path_prefix) tuples
        ag_count (int): Number of Ag atoms in the next generation
    
    Returns:
        new_structures (list): List of newly generated (Atoms object, path_prefix) tuples
    """
    new_structures = []
    directions = fibonacci_sphere(n_points)

    for atoms, path_prefix in tqdm(structures_with_path, desc=f"Expanding {ag_count}Ag"):
        ag_indices = [idx for idx, atom in enumerate(atoms) if atom.symbol == 'Ag']
        child_count = 0
        for anchor_idx in ag_indices:
            anchor_pos = atoms[anchor_idx].position
            for dir_idx, direction in enumerate(directions):
                new_pos = anchor_pos + bond_distance * direction
                if not is_valid_position(new_pos, atoms.get_positions(), min_dist_to_any_atom):
                    continue
                # Create new structure with added Ag atom
                new_atoms = atoms.copy()
                new_atoms += Atoms('Ag', positions=[new_pos])
                # Set output path
                new_path = f"{path_prefix}_{child_count}"
                folder = os.path.join(f"{ag_count}Ag", new_path)
                os.makedirs(folder, exist_ok=True)
                write(os.path.join(folder, 'POSCAR'), new_atoms, format='vasp', vasp5=True)
                new_structures.append((new_atoms, new_path))
                child_count += 1
    return new_structures

# ========== MAIN EXECUTION ==========
if not os.path.exists(poscar_path):
    raise FileNotFoundError(f"{poscar_path} not found")

# Step 1: Load initial structure and validate it has only 1 Ag atom
structure = read(poscar_path, format='vasp')
ag_atoms = [atom for atom in structure if atom.symbol == 'Ag']
if len(ag_atoms) != 1:
    raise ValueError("Initial POSCAR must contain exactly 1 Ag atom.")

# Save initial structure to 1Ag/0
folder_1Ag = "1Ag/0"
os.makedirs(folder_1Ag, exist_ok=True)
write(os.path.join(folder_1Ag, 'POSCAR'), structure, format='vasp', vasp5=True)
structures_with_path = [(structure, "0")]

# Step 2 to max_ag_atoms: Recursively generate structures with additional Ag atoms
for ag_count in range(2, max_ag_atoms + 1):
    print(f"Generating {ag_count}Ag structures...")
    structures_with_path = generate_next_ag_structures(structures_with_path, ag_count)
    print(f"Generated {len(structures_with_path)} structures for {ag_count}Ag.")

print("All structures generated.")
