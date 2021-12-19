from rdkit import Chem
import tensorflow as tf

from features import *


def molecules_from_smile(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeMol.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def graph_from_smile(smile_list):
    atom_feature_list = []
    bond_feature_list = []
    pair_feature_list = []

    for smile in smile_list:
        molecule = molecules_from_smile(smile)
        atom_features,bond_features, pair_indices = graph_from_molecule(molecule)

        atom_feature_list.append(atom_features)
        bond_feature_list.append(bond_features)
        pair_feature_list.append(pair_indices)

    return(
     tf.ragged.constant(atom_feature_list, dtype=tf.float32),  
     tf.ragged.constant(bond_feature_list, dtype=tf.float32),  
     tf.ragged.constant(pair_feature_list, dtype=tf.int64),   
    )