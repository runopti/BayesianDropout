import tensorflow as tf
import six
#tf.Session() # will wait for inputs from keyboard
#six.moves.input()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))

#x = np.linspace(-10,10, 100)
#y = x*np.sin(x)
#plt.plot(x,y)
#plt.show()
n_sample = 20
import random
#x = np.random.multivariate_normal(np.array([0]*n_sample), np.eye(n_sample))*10-5
#x = np.random.multivariate_normal(np.array([0]*n_sample), np.eye(n_sample))*10-5
x = np.random.uniform(0, 1, n_sample)*10-5 
y = x*np.sin(x)
plt.scatter(x,y)
plt.plot([1,2,3],[5,8,10])
#plt.show()



from utils import synthetic

syn = synthetic.xsinx(n_samples=100, seed=123)

plt.scatter(syn.train.X, syn.train.Y)
plt.show()
