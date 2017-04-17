import numpy as np
from collections import namedtuple
#import base

Dataset = namedtuple('Dataset','data target')  
Datasets = namedtuple('Datasets','train validation test')

class DataSet(object):
    def __init__(self, images, labels, dtype=np.float32):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        num = images.shape
        self._num_examples = num[0]

    @property
    def X(self):
        return self._images

    @property
    def Y(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]  


def xsinx(n_samples=100, noise=None, seed=None,  *args, **kwargs):
    """ Create points along with sin(x^p) curve. 
      Args:
	n_samples: int, number of datapoints to generate
	noise: float or None, standard deviation of the Gaussian noise added
	seed: int or None, seed for the noise
      Returns:
	Shuffled xs and ys for sin curve synthetic dataset of type `base.Dataset`
    """
    if seed is not None:
      np.random.seed(seed)
    def generate(n_samples): 
        x = np.random.uniform(0, 1, n_samples)*10-5 
        y = x*np.sin(x) 
        return (x[:,np.newaxis],y[:,np.newaxis])

    train_x, train_y = generate(n_samples)
    test_x, test_y = generate(n_samples)
    valid_x, valid_y = generate(n_samples)

    train = DataSet(train_x, train_y)
    test = DataSet(test_x, test_y)
    validation = DataSet(valid_x, valid_y)

    return Datasets(train=train, validation=validation, test=test)
