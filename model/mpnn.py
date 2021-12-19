import re
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.backend import shape


class NodeCentral(layers.Layer):
    def __init__(self, hid, steps, **kwargs):
        self.hid = hid
        self.steps = steps
        super().__init__()

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1], 
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(shape=(self.atom_dim, self.bond_dim * self.bond_dim),
                                            trainable=True, initializer='glorot_uniform'
                                            )
        self.bias = self.add_weight(shape=(self.bond_dim * self.atom_dim), 
                                        trainable=True, initializer='zeros'
                                        )

        self.weight_node = self.add_weight(shape=self.hid * (self.atom_dim + self.hid),
                                                trainable=True, initializer='glorot_uniform'
                                                )
        self.weight_node_inp = self.add_weight(shape=(self.hid * self.atom_dim), 
                                                    trainable=True, initializer='glorot_uniform'                
                                                    )

        self.built = True

    def call(self, inputs):
        atoms, bonds, pairs = inputs

        atoms = tf.matmul(atoms, self.kernel) + self.bias
        atoms = tf.reshape(atoms, shape=(-1, self.bond_dim, self.bond_dim))

        bond_neighbor = tf.gather(bonds, pairs[:, 1])
        bond_neighbor = tf.expand_dims(bond_neighbor, axis=-1)

        trans = tf.matmul(atoms, bond_neighbor)
        trans = tf.squeeze(trans, axis=-1)
        aggregate = tf.math.segment_sum(trans, pairs[:, 0])

        aggregateed_values = []
        for i in range(self.steps):
            nodes = tf.matmul(self.weight_node, aggregate)

            edges = tf.nn.relu(tf.matmul(self.weight_node_inp, bonds))
            result = tf.nn.relu(nodes + edges)
            aggregateed_values.append(result)
        return aggregateed_values


class EdgeCentral(layers.Layer):
    def __init__(self, hid, steps, **kwargs):
        self.hid = hid
        self.steps = steps
        super().__init__()

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1], 
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(shape=(self.atom_dim, self.bond_dim * self.bond_dim),
                                            trainable=True, initializer='glorot_uniform'
                                            )
        self.bias = self.add_weight(shape=(self.bond_dim * self.atom_dim), 
                                        trainable=True, initializer='zeros'
                                        )

        self.weight_edge = self.add_weight(shape=self.hid * (self.bond_dim + self.hid),
                                                trainable=True, initializer='glorot_uniform'
                                                )
        self.weight_edge_inp = self.add_weight(shape=(self.hid * self.bond_dim), 
                                                    trainable=True, initializer='glorot_uniform'                
                                                    )

        self.built = True
 
    def call(self, inputs):
        atoms, bonds, pairs = inputs

        atoms_hid = tf.matmul(atoms, self.kernel) + self.bias
        atoms_hid = tf.reshape(atoms, shape=(-1, self.bond_dim, self.atom_dim))

        bond_hid = tf.matmul(bonds, self.kernel) + self.bias
        bond_hid = tf.reshape(bonds, shape=(-1, self.bond_dim, self.atom_dim))

        features = tf.concat([atoms_hid, bond_hid, atoms], axis=1)

        bond_neighbor = tf.gather(bonds, pairs[:, 1])
        bond_neighbor = tf.expand_dims(bond_neighbor, axis=-1)

        aggregateed_values = []
        for i in range(self.steps):
            nodes = tf.matmul(self.weight_edge, features)

            edges = tf.nn.relu(tf.matmul(self.weight_edge_inp, bond_neighbor))
            result = tf.nn.relu(nodes + edges)
            aggregateed_values.append(result)
        return aggregateed_values
        