from operator import pos
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras

from distance import LPDistance


class TripleLoss:
    def __init__(self, threshold: float, margin: Optional[float] = None, 
                        soft: Optional[bool] = False, eta: Optional[float] = None):
        
        self.threshold = threshold
        self.margin = margin
        self.soft = soft
        self.eta = eta

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        x = LPDistance(normarlize_emb=False, p=2, power=1)
        emb_distance_matrix = x(embs)

        lpydist = LPDistance(normarlize_emb=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        positive_emb = tf.where(tf.less(y_distance_matrix, self.threshold), emb_distance_matrix, tf.Tensor(0. ))
        negative_emb = tf.where(y_distance_matrix >= self.threshold, emb_distance_matrix, tf.Tensor(0. ))

        n_positive_triplet = 0
        loss_loop = 0
        for i in range(tf.size(embs)):
            pos_i = positive_emb[i][positive_emb[i] > 0]
            neg_i = negative_emb[i][negative_emb[i] > 0]
            pairs = tf.meshgrid(pos_i, -neg_i)
            if self.soft:
                log_loss = tf.math.softplus(pairs)
                if self.eta is not None:
                    pos_y_i = y_distance_matrix[i][positive_emb[i] > 0]
                    neg_y_i = y_distance_matrix[i][negative_emb[i] > 0]
                    pairs_y = tf.meshgrid(pos_y_i, neg_y_i)
                    assert pairs.shape == pairs_y.shape, (pairs_y.shape, pairs.shape)
                    weight_pos = self.smooth_indicator(self.threshold - pairs_y[:, 0]) / self.smooth_indicator(self.threshold)
                    weight_neg = self.smooth_indicator(pairs_y[:, 1] - self.threshold) / self.smooth_indicator(1 - self.threshold)
                    triplet_loss = log_loss * weight_pos * weight_neg
                
            else:
                triplet_loss = tf.nn.relu(self.margin + np.sum(pairs, axis=-1))
                
            n_positive_triplet += np.sum(triplet_loss > 0)
            loss_loop += np.sum(triplet_loss)

        loss_loop = loss_loop / max(1, n_positive_triplet)
        return loss_loop

    def smooth_indicator(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, float):
            return np.tanh(x / (2 * self.eta))
        return tf.tanh(x / (2 * self.eta))

    def compute_loss(self, embs: Tensor, ys: Tensor):
        return self.build_loss_matrix(embs, ys)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id(threshold: float, margin: Optional[float] = None, 
                        soft: Optional[bool] = None, eta: Optional[bool] = None) -> str:
        
        if margin is not None:
            return f'triplet-thr-{threshold: g}-mrg--{margin:g}'
        if soft is not None:
            metric_id = f'triplet-thr-{threshold: g}-soft'
            if eta is not None:
                metric_id += f'-eta-{eta:g}'
            return metric_id

if __name__ == '__main__':
    mod = TripleLoss(threshold=1.0, margin=0.2,soft=True, eta=1)
    x = keras.layers.Input(shape=(64, ))
    y = keras.layers.Input(shape=(50, ))
    
    print(mod(x, y))