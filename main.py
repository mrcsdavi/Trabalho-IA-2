from twoCircles import TwoCircles
from twoMoons import TwoMoons
from iris import Iris
from algoritmo import Algoritmo
from plot import plot_dbscan

from sklearn.datasets import make_circles, make_moons

def main():


    # TWO CIRCLES (DBSCAN)
    Xc, y = make_circles(
    n_samples=300,
    factor=0.5,
    noise=0.05,
    random_state=None
    )

    clusters, ruido, tipos = Algoritmo.dbscan(
        Xc, eps=0.15, min_pts=5
    )

    plot_dbscan(Xc, tipos, "DBSCAN - Two Circles")


    # TWO MOONS (DBSCAN)
    Xm, y = make_moons(
    n_samples=300,
    noise=0.08,
    random_state=None   
    )

    clusters, ruido, tipos = Algoritmo.dbscan(
        Xm, eps=0.18, min_pts=5
    )

    plot_dbscan(Xm, tipos, "DBSCAN - Two Moons")

    # IRIS
    X, y, names = Iris.iris()
    X2 = X[:, [2, 3]]

    tipos = Algoritmo.dbscan(
        X2, eps=0.5, min_pts=5
    )

    plot_dbscan(X2, tipos, "DBSCAN - Iris (Petal)")

if __name__ == "__main__":
    main()
