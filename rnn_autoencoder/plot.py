import matplotlib.pyplot as plt
import numpy as np
import time

def plot(loss, iterations):
    plt.plot(iterations, loss)
    plt.title(f'Evolution of loss during iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(f'result_pics/plot_loss_{time.time()}.png', bbox_inches='tight')		
    plt.cla()


def scatter_plot(data):
    plt.scatter(data[0], data[1])
    plt.title(f'Words in the output sentence vs ROUGE-1 F-score')
    plt.xlabel('ROUGE-1 F-score')
    plt.xlabel('Number of words in the output')
    plt.savefig(f'result_pics/plot_scatter_{time.time()}.png', bbox_inches='tight')		
    plt.cla()