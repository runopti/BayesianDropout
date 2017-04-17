import tensorflow as tf
import numpy as np
import math
import os

class Config(object):
    def __init__(self):
        self.sample_size = 20
        self.n_input = 1
        self.n_h1 = 20
        self.n_h2 = 20
        self.n_output = 1
        self.batch_size = 1
        self.lr = 0.01
        # for dropout
        self.p_list_train = [0.95, 0.95]
        self.p_list_validation = [1,1]
        

class MLP(object):
    def __init__(self, config):
        self.input = tf.placeholder(tf.float32, shape=[config.batch_size, config.n_input])
        self.target = tf.placeholder(tf.float32, shape=[config.batch_size, config.n_output])
        self.keep_prob_h1 = tf.placeholder(tf.float32)
        self.keep_prob_h2 = tf.placeholder(tf.float32)
            
        # feed the data to the model and get the output 
        self.output = self._inference(self.input, config)
        self.loss = tf.reduce_mean(tf.square(self.output - self.target))
        # For backprop call
        self.train_op = tf.train.GradientDescentOptimizer(config.lr).minimize(self.loss)

        # For convenience
        self.y_hat = self.output #tf.nn.softmax(self.logits)
        self.acc = self.loss

        num_params = 0
        for var in tf.global_variables():
            num_params += np.prod(var.get_shape().as_list())
        print("Number of Model Parameters: {}".format(num_params))
        print("Size of Model : {} MB".format(num_params*4/1e6)) #4 = tf.float32

        # Create a saver
        self.saver = tf.train.Saver()

        # Start Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _inference(self, input, config):
        h1 = self._add(input, config.n_input, config.n_h1, "ReLU", "hidden1")
        h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob_h1)
        h2 = self._add(h1, config.n_h1, config.n_h2, "Sigmoid", "hidden2")
        h2 = tf.nn.dropout(h2, keep_prob=self.keep_prob_h2)
        output = self._add(h2, config.n_h2, config.n_output, "Linear", "hidden3")
        return output

    def _add(self, prev, n_in, n_out, activation, name_scope):
        with tf.name_scope(name_scope) as scope:
            weights = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0/math.sqrt(float(n_in)), name="weights"))
            biases = tf.Variable(tf.zeros([n_out]), name="biases") 
            #weights = tf.get_variable("weights", [n_in, n_out], tf.random_normal_initializer())
            #biases = tf.get_variable("biases", [n_out], tf.random_normal_initializer())
            if activation=="ReLU":
                hidden = tf.nn.relu(tf.matmul(prev, weights) + biases)
                return hidden
            elif activation=="Sigmoid":
                hidden = tf.nn.sigmoid(tf.matmul(prev, weights) + biases)
                return hidden
            elif activation=="Linear":
                hidden = tf.matmul(prev, weights) + biases
                return hidden
            elif activation=="Softmax":
                # softmax will be done in the loss calc by tf.nn.softmax_cross_entropy_with_logits
                hidden = tf.matmul(prev, weights) + biases
                return hidden 
            else:
                print("Didn't specify activation!!!")
                raise NotImplementedError()
            

    def forward_backprop(self, data, targets, p):
        cost, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input:data, self.target: targets, self.keep_prob_h1: p[0], self.keep_prob_h2: p[1]})
        return cost

    def forward(self, data, targets, p):
        cost, acc = self.sess.run([self.loss, self.acc], feed_dict={self.input:data, self.target: targets, self.keep_prob_h1: p[0], self.keep_prob_h2: p[1]})
        return cost, acc

    def get_prediction(self, data, p):
        y_hat = self.sess.run([self.y_hat], feed_dict={self.input: data, self.keep_prob_h1: p[0], self.keep_prob_h2: p[1]})
        return y_hat[0]

    def get_accuracy(self, data, targets):
        acc = self.sess.run([self.acc], feed_dict={self.input:data, self.target: targets})
        return acc[0]

    def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)
        

    def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)
              
    def getWeights(self):
        pass

