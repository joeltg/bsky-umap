import sys

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    arguments = sys.argv[1:]
    input_path = arguments[0]

    data = np.load(input_path)

    plt.hist(data, bins=500)
    plt.show()
