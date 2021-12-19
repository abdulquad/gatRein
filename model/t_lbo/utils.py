import tensorflow as tf
import numpy as np


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def add_to_recordable_attributes(
    input_obj, name=None, list_of_names=None, is_stat=False
    ):
    if is_stat:
        attr_name_list = '_record_these_stats'
    else:
        attr_name_list = '_record_these'
    if not hasattr(input_obj, attr_name_list):
        setattr(input_obj,attr_name_list, [])
    attr_name_ = getattr(input_obj, attr_name_list)
    if name is not None:
        if name not in attr_name_:
            attr_name_.append(name)
        if not hasattr(input_obj, name):
            setattr(input_obj, name, 0)
    if list_of_names is not None and is_list_or_tuple(list_of_names):
        for n in list_of_names:
            add_to_recordable_attributes(input_obj, name=n, is_stat=is_stat)


def reset_stats(input_obj):
    for attr_list in ['_record_these_stats']:
        for r in getattr(input_obj, attr_list, []):
            setattr(input_obj, r, 0)

def meshgrid_from_sizes(x, y):
    a = np.arange(tf.size(x))
    b = np.arange(tf.size(y))
    return tf.meshgrid(a, b)

class ModuleWithRecord(tf.keras.layers.Layer):
    def __init__(self, collect_stats=False):
        self.collect_stats = collect_stats
        super().__init__()

    def add_to_recordable_attr(self, name=None, list_of_names=None, is_stat=False):
        if is_stat and not self.collect_stats:
            pass
        else:
            add_to_recordable_attributes(self, name, list_of_names, is_stat)

    def reset_stat(self):
        reset_stats(self)