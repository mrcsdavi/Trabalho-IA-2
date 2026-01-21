from sklearn.datasets import load_iris

class Iris:
    def load_data(self):
        iris = load_iris()
        return iris.data, iris.target, iris.target_names