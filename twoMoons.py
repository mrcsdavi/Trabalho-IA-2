import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class TwoMoons:
    def twoMoons():
        X, y = make_moons(n_samples=300, noise=0.08, random_state=42)
        
        # plot
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.title("Dataset: Two Moons")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()