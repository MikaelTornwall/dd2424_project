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