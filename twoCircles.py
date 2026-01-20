import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

class TwoCircles:
    def twoCircles():

        # gera os dados
        X, y = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

        # plot
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.title("Dataset: Circles")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

