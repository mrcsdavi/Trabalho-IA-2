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

    # IRIS DATASET (DBSCAN)
    iris_loader = Iris()
    X_iris, y_true, target_names = iris_loader.load_data()
    
    # Aplicar DBSCAN no Iris
    clusters_iris, ruido_iris, tipos_iris = Algoritmo.dbscan(
        X_iris, eps=0.5, min_pts=5  # Valores de eps e min_pts podem precisar de ajuste
    )
    
    print(f"\nIris Dataset Results:")
    print(f"Número de clusters encontrados: {len(clusters_iris)}")
    print(f"Número de pontos considerados ruído: {ruido_iris}")
    print(f"Target names: {target_names}")
    
    plot_dbscan(X_iris[:, :2], tipos_iris, "DBSCAN - Iris")

if __name__ == "__main__":
    main()