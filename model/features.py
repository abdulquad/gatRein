import numpy as np

class Featurizer:
    def __init__(self, allowable_set):
        self.dims = 0
        self.feature_mapping = {}
        for k, s in allowable_set.items():
            s = sorted(list(s))
            self.feature_mapping[k] = dict(zip(s, range(self.dims, len(s) + self.dims)))
            self.dims += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dims, ))
        for name_feature, feature_mapping in self.feature_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_set):
        super().__init__(allowable_set)
    
    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valency(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom. GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_set):
        super.__init__(allowable_set)
        self.dims += 1

    def encode(self, bond):
        output = np.zeros((self.dims, ))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output


atom_featurizer = AtomFeaturizer(
    allowable_set= {
        'symbol': {'B', 'Br', 'C', 'Ca', 'Cl', 'F', 'H', 'I', 'N', 'Na', 'O', 'P', 'S'},
        'n_valency': {0, 1, 2, 3, 4, 5, 6},
        'n_hydrogens': {0, 1, 2, 3, 4},
        'hybridization': {'s', 'sp', 'sp2', 'sp3'},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_set={
        'bond_type': {'single', 'double', 'triple', 'aromatic'},
        'conjugated': {True, False},
    }
)