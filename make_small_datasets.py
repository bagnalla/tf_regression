import gzip, pickle, os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
def make_dataset(images, labels):
    return DataSet(images, labels, reshape=False, dtype=tf.uint8)

SIZE = 1

small_train_images = np.random.random([50000, SIZE])
small_validation_images = np.random.random([10000, SIZE])
small_test_images = np.random.random([10000, SIZE])

small_train_labels = np.random.random_integers(0, 9, size=50000)
small_validation_labels = np.random.random_integers(0, 9, size=10000)
small_test_labels = np.random.random_integers(0, 9, size=10000)

small_train = make_dataset(small_train_images, small_train_labels)
small_validation = make_dataset(small_validation_images, small_validation_labels)
small_test = make_dataset(small_test_images, small_test_labels)

print(small_train_images.shape)
print(small_validation_images.shape)
print(small_test_images.shape)

os.makedirs('small/', exist_ok=True)
with gzip.open('small/train.pkl.gz', 'w') as f:
    pickle.dump(small_train, f)
with gzip.open('small/validation.pkl.gz', 'w') as f:
    pickle.dump(small_validation, f)
with gzip.open('small/test.pkl.gz', 'w') as f:
    pickle.dump(small_test, f)

print(small_train_images)
print(small_train_labels)

# For compile.py script
with gzip.open('small/test_images.pkl.gz', 'w') as f:
    pickle.dump(small_test_images, f)
