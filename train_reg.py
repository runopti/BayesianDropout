import tensorflow as tf
import os,sys,math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from utils import synthetic 
from args import args
from models.mlp_reg import MLP, Config
from utils.progress import Progress
import pickle 

def train_net(synthetic_data, max_epoch, retrain=None):
    # load data
    ## we can access to raw data like this:
    ## x = synthetic_data.train.X;  images.shape = []
    ## y = synthetic_data.train.Y; each label is a probability distribution.
    
    with tf.Graph().as_default():
        config = Config()
        
        if retrain==None:
            model = MLP(config)
        else:
            model = retrain
        tf.get_default_graph().finalize() 

        progress = Progress()

        n_batch_loop = int(synthetic_data.train.num_examples/config.batch_size)
        for epoch in range(max_epoch):
            sum_cost = 0
            progress.start_epoch(epoch, max_epoch)

            for t in range(n_batch_loop):
                # batch_X: batch_size x n_input
                # batch_y: batch_size
                batch_X, batch_y = synthetic_data.train.next_batch(config.batch_size)
                cost_per_sample = model.forward_backprop(batch_X, batch_y, config.p_list_train)
                sum_cost += cost_per_sample

                if t % 10 == 0:
                    progress.show(t, n_batch_loop, {})

            model.save(epoch, args.model_dir)
            
        
            # Validation
            val_loss, val_acc = evaluate(model, synthetic_data.validation, config)
            progress.show(n_batch_loop, n_batch_loop, {
                "val_loss" : val_loss,
                "val_acc" : val_acc,
                })

    return model

def evaluate(model, dataset, config):
    n_batch_loop = int(dataset.num_examples/config.batch_size)
    sum_cost = 0
    sum_acc = 0
    for t in range(n_batch_loop):
        batch_X, batch_y = dataset.next_batch(config.batch_size)
        cost_per_sample, acc = model.forward(batch_X, batch_y, config.p_list_validation)
        sum_cost += cost_per_sample 
        sum_acc += acc 
    acc_avg = sum_acc / n_batch_loop
    cost_avg = sum_cost / n_batch_loop
    return cost_avg, acc_avg


def test():
    pass

if __name__=="__main__":
    train_net()
    #test()
