import numpy as np
import matplotlib.pyplot as plt
from utils import synthetic
import train_reg
#from collections import namedtuple

canvas_size = (-10, 10)
n_test_point = 20
n_vis_point = 100
T = 10


def draw():
    new_data = synthetic.xsinx(n_samples=n_test_point)

    fig, ax = plt.subplots(1)
    net = train_reg.train_net(new_data, max_epoch=1)
    x, y_mean, y_std = get_mean(net)
    lines = ax.plot(x,y_mean) 

    while True:
        # update data to be plotted
        net = train_reg.train_net(new_data, max_epoch=1, retrain=net)
        x, y_mean, y_std = get_mean(net)

	# delete the previous data on the plot
        for coll in (ax.collections):
            ax.collections.remove(coll)
	# Update the plot
        lines[0].set_data(x, y_mean)
        ax.set_xlim([-10, 10])
        ax.set_ylim([-15, 5])
        ax.scatter(new_data.train.X, new_data.train.Y)
        for i in range(2):
            ax.fill_between(x, y_mean+((i+1))*y_std, y_mean-((i+1))*y_std, facecolor='blue', alpha=0.125*(i+1))
        plt.pause(.001)
    
def test():
    new_data = synthetic.xsinx(n_samples=n_test_point)
    net = train_reg.train_net(new_data)

def get_mean(net):
    x = np.linspace(canvas_size[0],canvas_size[1],n_vis_point)
    # Refactor these loops later
    y = np.empty([T, n_vis_point])
    for t in range(T):
        for i, x_i in enumerate(x):
            y[t,i] = net.get_prediction(np.array(x_i).reshape(1,1), p=[0.8, 0.8])
    
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    return (x, y_mean, y_std)

if __name__=='__main__':
    #test()
    draw()
