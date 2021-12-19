import tensorflow as tf
import neural_structured_learning as nsl

from utils import ModuleWithRecord, meshgrid_from_sizes


class LPDistance(ModuleWithRecord):
    def __init__(self, normarlize_emb=True, p=2, power=1, is_inverted=False, **kwargs):
        self.normalize_emb = normarlize_emb
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attr(list_of_names=['p', 'power'], is_stat=False)

    def call(self, query_emb, ref_emb=None):
        self.reset_stat()
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)

        self.set_default_stats(query_emb, ref_emb, query_emb_normalized, ref_emb_normalized)
        mat = self.comput_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        assert tf.size(mat) == tf.size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        dtype = query_emb.dtype
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == tf.float16:
            rows, cols = meshgrid_from_sizes(query_emb, ref_emb)
            output = tf.zeros(tf.size(rows), dtype)
            rows, cols = tf.nest.flatten(rows), tf.nest.flatten(cols)
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output

    def pairwise_distance(self, query_emb, ref_emb):
        val = nsl.keras.layers.PairwiseDistance(distance_config=-1)
        return val(sources=query_emb, targets=ref_emb)

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return tf.math.maximum(*args, **kwargs)
        return tf.math.minimum(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return tf.math.minimum(*args, **kwargs)
        return tf.math.maximum(*args, **kwargs)

    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embs, dim, **kwargs):
        return tf.linalg.normalize(embs, self.p, dim, **kwargs)

    def maybe_normalize(self, embs, dim=1, **kwargs):
        if self.normalize_emb:
            return self.normalize(embs, dim=dim, **kwargs)
        return embs

    def get_norm(self, embs, dim=1, **kwargs):
        return tf.norm(embs, self.p, dim, **kwargs)

    def set_default_stats(self, query_emb, ref_emb, query_emb_norms, ref_emb_norms):
        if self.collect_stats:
            with tf.no_gradient():
                stats_dict = {
                    'initial_avg_query_norm': tf.keras.metrics.Mean(self.get_norm(query_emb)).item(),
                    'initial_avg_ref_norm': tf.keras.metrics.Mean(self.get_norm(ref_emb)).item(),
                    'final_avg_query_norm': tf.keras.metrics.Mean(self.get_norm(query_emb_norms)).item(),
                    'final_avg_ref_norm': tf.keras.metrics.Mean(self.get_norm(ref_emb_norms)).item(),
                }
                self.set_stats(stats_dict)

    def set_stats(self, stats_dict):
        for k,v in stats_dict.items():
            self.add_to_recordable_attr(name=k, is_stat=True)
            setattr(self, k, v)