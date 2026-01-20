import matplotlib.pyplot as plt

def plot_dbscan(X, tipos, titulo):
    cores = {
        "core": "blue",
        "border": "green",
        "noise": "red"
    }

    plt.figure()
    for t in ["core", "border", "noise"]:
        idx = [i for i in range(len(X)) if tipos[i] == t]
        plt.scatter(X[idx, 0], X[idx, 1],
                    c=cores[t], label=t)

    plt.title(titulo)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()
