# This file provides dataset-specific parameters and functions for
# MNIST and EMNIST.

import gzip, os, pickle
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

def make_dataset(images, labels):
    return DataSet(images, labels, reshape=False, dtype=tf.uint8)


# RANDOM

RANDOM_INPUT_SIZE = 1
import regression_model as regression_model

def random_example_shape(batch_size):
    return (batch_size, RANDOM_INPUT_SIZE)

def random_load_data():
    with gzip.open('small/train.pkl.gz', 'rb') as f:
        train = pickle.load(f)
    with gzip.open('small/validation.pkl.gz', 'rb') as f:
        validation = pickle.load(f)
    with gzip.open('small/test.pkl.gz', 'rb') as f:
        test = pickle.load(f)
    return train, validation, test

def choose_dataset(set_name):
    if set_name.lower() == 'random':
        return regression_model, RANDOM_INPUT_SIZE, \
            random_example_shape, random_load_data
    else:
        return None
