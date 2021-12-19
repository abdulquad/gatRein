import tensorflow as tf
from tensorflow import keras as k

class GraphAttention(k.layers.Layer):
    def __init__(self, unit, **kwargs):
        self.unit = unit
        self.initializer='glorot_uniform'
        self.regularizer = 'zeros'
        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.unit),
                                    trainable=True,
                                    initializer=self.initializer,
                                    regularizer=self.regularizer
                                    )
        self.kernel_weight = self.add_weight(
                                    shape=(self.unit * 2, 1),
                                    trainable=True,
                                    initializer=self.initializer,
                                    regularizer=self.regularizer
                                    )

        self.build = True

    def call(self, inputs):
        nodes, edges = inputs
        node_features = tf.matmul(nodes, self.kernel)
        edge_features = tf.matmul(edges, self.kernel)
        
        features = tf.concat([node_features, edge_features])
        features = tf.matmul(tf.transpose(self.kernel_weight), features)

        values = tf.nn.leaky_relu(features)

        attention_score = tf.squeeze(values, -1)

        attention_score = tf.math.exp(tf.clip_by_value(attention_score, -2, 2))
        attention_score_sum = tf.math.unsorted_segment_sum(
            data=attention_score,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0], keepdims=True),
        )
        attention_score_sum = tf.repeat(attention_score_sum, 
                                    tf.math.bincount(tf.cast(edges[:, 0], 'int32')))

        attention_norm = attention_score / attention_score_sum
        
        node_neigb = tf.gather(node_features, edges[:, 1])
        output = tf.math.unsorted_segment_sum(
            data=node_neigb * attention_norm[:, tf.newaxis], 
            segment_ids=edges[:, 0],
            num_segments=tf.shape(nodes)[0]
        )

        return output


class MultiHeadAttention(k.layers.Layer):
    def __init__(self, units, num_heads=8, merge_type='concat', **kwargs):
        super().__init__(**kwargs)

        self.attention_layer=[GraphAttention(unit=units) for _ in range(num_heads)]
        self.merge_type = merge_type
        self.num_heads = num_heads

    def call(self, inputs):
        atoms, indicies = inputs

        outputs = [attention([atoms, indicies]) for attention in self.attention_layer]

        if self.merge_type =='concat':
            outputs = tf.concat(outputs, axis=-1)

        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

        return tf.nn.relu(outputs)