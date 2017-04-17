import numpy as np
import matplotlib.pyplot as plt
from utils import synthetic
import train_reg
#from collections import namedtuple

canvas_size = (-10, 10)
n_test_point = 20
n_vis_point = 300
T = 10


def draw():
    new_data = synthetic.xsinx(n_samples=n_test_point)

    net = train_reg.train_net(new_data)
    x, y_mean, y_std = get_mean(net)

    fig, ax = plt.subplots(1)
    #plt.figure(figsize=(8,4))
    ax.scatter(new_data.train.X, new_data.train.Y)
    ax.plot(x,y_mean) 
    ax.fill_between(x, y_mean+y_std, y_mean-y_std, facecolor='blue', alpha=0.5)
    ax.fill_between(x, y_mean+2*y_std, y_mean-2*y_std, facecolor='blue', alpha=0.25)
    plt.show()

    
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
