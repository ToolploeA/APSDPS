from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# !!! counting start from 0 !!!
porphyrin_N_atom_idx = [382, 387, 377, 392]
Mg_atom_idx = [291]
O_atom_idx = [88, 89, 195, 196, 210, 211]
N_atom_idx = [64]

def get_atom_with_idx(mol, idx_list):
    return [mol.GetAtomWithIdx(i) for i in idx_list]

def r_region_check(filename: str):
    '''
    Check if the struc should be in the R-region

    Parameters
    ----------
    filename: str
        The filename of the xyz file
    
    Returns
    -------
    bool
        whether in the R-region
    '''
    mol = Chem.MolFromXYZFile(filename)
    rdDetermineBonds.DetermineConnectivity(mol, useVdw = True, covFactor = 1.1)

    condition = []

    Mg_atom = get_atom_with_idx(mol, Mg_atom_idx)
    porphyrin_N_atom = get_atom_with_idx(mol, porphyrin_N_atom_idx)
    his_N_atom = get_atom_with_idx(mol, N_atom_idx)
    O_atom = get_atom_with_idx(mol, O_atom_idx)

    atom_idx_bond_to_Mg = [bond.GetOtherAtom(Mg_atom[0]).GetIdx() for bond in Mg_atom[0].GetBonds()]
    num_bond_to_porphyrin_N = [len(atom.GetBonds()) for atom in porphyrin_N_atom]
    atom_symbol_bond_to_his_N = [bond.GetOtherAtom(his_N_atom[0]).GetSymbol() for bond in his_N_atom[0].GetBonds()]
    atom_symbol_bonded_to_O = [[bond.GetOtherAtom(O_atom[idx]).GetSymbol() for bond in bond_list] for idx, bond_list in enumerate([atom.GetBonds() for atom in O_atom])] # the element symbol of the atom bonded with the 4 Glu O atoms, list[4]
    O_idx_bond_with_H = [i for idx, i in enumerate(O_atom_idx) if 'H' in atom_symbol_bonded_to_O[idx]]


    condition.append(all(i in atom_idx_bond_to_Mg for i in porphyrin_N_atom_idx)) # Mg in the center of porphyrin ring, bonded with all 4 N atoms
    condition.append(all(item == 3 for item in num_bond_to_porphyrin_N)) # no other connection on the porphyrin ring N
    combined_idx = O_atom_idx + N_atom_idx
    condition.append(any(i in atom_idx_bond_to_Mg for i in combined_idx)) # Glu coordinated with Mg atom
    condition.append('H' not in atom_symbol_bond_to_his_N) # no HIP
    condition.append(not (set(atom_idx_bond_to_Mg) & set(O_idx_bond_with_H))) # Mg-coord O atom has no H

    return all(condition)
    # return False

def p_region_check(filename: str):
    '''
    Check if the struc should be in the P-region

    Parameters
    ----------
    filename: str
        The filename of the xyz file
    
    Returns
    -------
    bool
        whether in the P-region
    '''
    mol = Chem.MolFromXYZFile(filename)
    rdDetermineBonds.DetermineConnectivity(mol, useVdw = True, covFactor = 1.1)

    condition = []

    Mg_atom = get_atom_with_idx(mol, Mg_atom_idx)
    porphyrin_N_atom = get_atom_with_idx(mol, porphyrin_N_atom_idx)

    atom_idx_bond_to_Mg = [bond.GetOtherAtom(Mg_atom[0]).GetIdx() for bond in Mg_atom[0].GetBonds()]
    bond_to_porphyrin_N = [atom.GetBonds() for atom in porphyrin_N_atom] # the bond which bonded with the 4 porphyrin N atoms, list[4]
    atom_symbol_bonded_to_porphyrin_N = [[bond.GetOtherAtom(porphyrin_N_atom[idx]).GetSymbol() for bond in bond_list] for idx, bond_list in enumerate(bond_to_porphyrin_N)] # the element symbol of the atom bonded with the 4 porphyrin N atoms, list[4]

    condition.append(all(i not in atom_idx_bond_to_Mg for i in porphyrin_N_atom_idx)) # Mg not bonded with porphyrin N
    condition.append(sum(sublist.count('H') == 1 for sublist in atom_symbol_bonded_to_porphyrin_N) == 2) # 2 H bonded in the porphyrin N
    condition.append(sum([i in atom_idx_bond_to_Mg for i in O_atom_idx]) >= 3) # Glu coordinated with Mg atom

    return all(condition)
    # return False

def i_region_check(filename: str, template_xyz: str) -> bool:
    '''
    Check if the struc should be in the P-region

    Parameters
    ----------
    filename: str
        The filename of the xyz file
    
    Returns
    -------
    bool
        whether is a valid struc
    '''
    mol = Chem.MolFromXYZFile(filename)
    rdDetermineBonds.DetermineConnectivity(mol, useVdw = True)

    template_mol = Chem.MolFromXYZFile(template_xyz)
    rdDetermineBonds.DetermineConnectivity(template_mol, useVdw = True)

    condition = []

    C_atom = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    template_C_atom = [atom for atom in template_mol.GetAtoms() if atom.GetSymbol() == 'C']
    O_atom = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O']
    bond_to_O = [atom.GetBonds() for atom in O_atom]
    atom_symbol_bond_to_O = [[bond.GetOtherAtom(O_atom[idx]).GetSymbol() for bond in bond_list] for idx, bond_list in enumerate(bond_to_O)]
    H_atom = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    bond_to_H = [atom.GetBonds() for atom in H_atom]
    atom_symbol_bond_to_H = [[bond.GetOtherAtom(H_atom[idx]).GetSymbol() for bond in bond_list] for idx, bond_list in enumerate(bond_to_H)]

    # C_atom_num_H = [[a.GetSymbol() for a in atom.GetNeighbors()].count('H') for atom in C_atom]
    # template_C_atom_num_H = [[a.GetSymbol() for a in atom.GetNeighbors()].count('H') for atom in template_C_atom]
    C_bond_symbol = [[a.GetSymbol() for a in atom.GetNeighbors() if a.GetSymbol() != 'Mg'] for atom in C_atom]
    template_C_bond_symbol = [[a.GetSymbol() for a in atom.GetNeighbors() if a.GetSymbol() != 'Mg'] for atom in template_C_atom]

    condition.append(C_bond_symbol == template_C_bond_symbol) # C-X
    condition.append(not any('O' in x for x in atom_symbol_bond_to_O)) # no O-O bond
    condition.append(not any('H' in x for x in atom_symbol_bond_to_H)) # no H-H bond
    condition.append(['H', 'H', 'H'] not in atom_symbol_bond_to_O) # no H3O+

    return all(condition)