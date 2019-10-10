import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

digitData=mnist.load_data()

plt.imshow(digitData[0], cmap='Greys')

