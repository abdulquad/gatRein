from typing import List, Any, Callable, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from t_lbo.metrics import TripleLoss


class VAE(keras.layers.Layer):
    def __init__(self, beta_metric, threshold, margin, soft=True):
        
        self.threshold = threshold
        self.margin = margin
        self.soft = soft
        self.beta_metric = beta_metric
        self.eta = 0.1
        self.beta_final = 1e-4  
        self.beta_initial = 1e-6
        self.num_atoms = 9  
        self.atom_dim = 4 + 1  # Number of atom types
        self.bond_dim = 4 + 1  # Number of bond types
        self.latent_dim = 64
        self.gconv_units=[128, 128, 128, 128]
        super(VAE, self).__init__()

       
    def encode(self, latent_dim ,dense_units, dropout_rate, adjacency_shape, feature_shape):
        z = keras.layers.Input(shape=(latent_dim,))

        x = z
        for unit in dense_units:
            x = keras.layers.Dense(unit, activation='tanh')(x)
            x = keras.layers.Dropout(dropout_rate)(x)

        x_adjacency = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
        x_adjacency = keras.layers.Reshape(adjacency_shape)(x_adjacency)
        # Symmetrify tensors in the last two dimensions
        x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2
        mu = keras.layers.Softmax(axis=1)(x_adjacency)

        # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
        x_features = keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
        x_features = keras.layers.Reshape(feature_shape)(x_features)
        log_var = keras.layers.Softmax(axis=2)(x_features)

        return mu, log_var

    def decode(self,dense_units, gconv_units, dropout_rate,adjacency_shape, feature_shape):
        adjacency = keras.layers.Input(shape=adjacency_shape)
        features = keras.layers.Input(shape=feature_shape)

        # Propagate through one or more graph convolutional layers
        features_transformed = features
        for units in gconv_units:
            features_transformed = RelationalGraphConvLayer(units)(
                [adjacency, features_transformed]
            )
        # Reduce 2-D representation of molecule to 1-D
        x = keras.layers.GlobalAveragePooling1D()(features_transformed)

        # Propagate through one or more densely connected layers
        for units in dense_units:
            x = keras.layers.Dense(units, activation="relu")(x)
            x = keras.layers.Dropout(dropout_rate)(x)

        recon = keras.layers.Dense(1, dtype="float32")(x)
        return recon

    def sample_latent(self, mu, logvar):
        scale = tf.exp(logvar) + 1e-10
        dist = tfp.distributions.Normal(loc=mu, scale=scale)
        z_sample = dist.rsample()
        return z_sample

    def rec_loss(self,x , z):
        rec = self.decode(z)
        recon_loss = tf.reduce_mean(tf.reduce_sum(
                                    keras.losses.binary_crossentropy(x, rec), axis=(1, 2)
                                ))
        return recon_loss

    def kl_loss(self, mu, log_var):
        kl = -0.5 * (1 + log_var, tf.square(mu) - tf.exp(log_var))
        return tf.reduce_mean(tf.reduce_sum(kl, axis=1))

    def call(self, inputs, validation: bool = False):
        # remparameterize
        mu, log_var = self.encode(inputs=inputs)
        z_sample = self.sample_latent(mu, log_var)

        kl_loss = self.kl_loss(mu, log_var)
        rec = self.rec_loss(inputs, z_sample)
        
        if validation:
            beta = self.beta_final
        else:
            beta = self.beta_initial
        
        metric_loss = TripleLoss(
                                threshold=self.threshold,
                                margin=self.margin,
                                soft=self.soft, 
                                eta=self.eta
                )

        loss = rec + beta * kl_loss + self.beta_metric * metric_loss

        return loss

class RelationalGraphConvLayer(keras.layers.Layer):
    def __init__(
        self,
        units=128,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]

        self.kernel = self.add_weight(
            shape=(bond_dim, atom_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="W",
            dtype=tf.float32,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(bond_dim, 1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="b",
                dtype=tf.float32,
            )

        self.built = True

    def call(self, inputs, training=False):
        adjacency, features = inputs
        # Aggregate information from neighbors
        x = tf.matmul(adjacency, features[:, None, :, :])
        # Apply linear transformation
        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
        # Reduce bond types dim
        x_reduced = tf.reduce_sum(x, axis=1)
        # Apply non-linear transformation
        return self.activation(x_reduced)

if __name__ == '__main__':
    model = VAE(0.2, 1, 0.2)
    num_atoms = 9  
    atom_dim = 4 + 1  # Number of atom types
    bond_dim = 4 + 1  # Number of bond types
    latent_dim = 64

    enc = model.decode( 
        dense_units=[64, 128, 256],
        gconv_units=[512, 512, 512],
        dropout_rate=0.2,
        adjacency_shape=(bond_dim, num_atoms, num_atoms),
        feature_shape=(num_atoms, atom_dim),
        )

    print(enc)