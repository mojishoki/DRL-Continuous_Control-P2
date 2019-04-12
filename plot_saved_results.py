import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def plot_results(scores_filepath,network_name):
    """ 
        `scores_filepath` : file path where scores are saved
        `network_name` : the name of the network to be displayed on the plot
    """
    scores=np.loadtxt(scores_filepath)
    avgs=np.array([scores[max(i-100,0):i].mean() for i in range(1,len(scores)+1)])
    for i in range(100,len(avgs),100):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, avgs[i-1]))
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(len(avgs), avgs[-1]))
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(len(avgs), avgs[-1]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label=network_name)
    plt.plot(np.arange(len(avgs)), avgs, c='r', label='average')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left')
    plt.show()