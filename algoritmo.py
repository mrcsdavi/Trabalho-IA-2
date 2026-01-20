import math

class Algoritmo:

    @staticmethod
    def dist(p, q):
        return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(len(p))))

    @staticmethod
    def vizinhanca_eps(D, idx, eps):
        return [
            i for i in range(len(D))
            if Algoritmo.dist(D[idx], D[i]) <= eps
        ]

    @staticmethod
    def dbscan(D, eps, min_pts):
        visitado = set()
        cluster = {}           # índice -> id do cluster
        ruido = set()
        tipos = [""] * len(D) # core, border, noise
        C = 0

        for i in range(len(D)):
            if i in visitado:
                continue

            visitado.add(i)
            N = Algoritmo.vizinhanca_eps(D, i, eps)

            if len(N) < min_pts:
                ruido.add(i)
                tipos[i] = "noise"
            else:
                C += 1
                Algoritmo.expandir_cluster(
                    D, i, N, C, eps, min_pts,
                    visitado, cluster, ruido, tipos
                )

        return cluster, ruido, tipos

    @staticmethod
    def expandir_cluster(D, idx, N, C, eps, min_pts,
                         visitado, cluster, ruido, tipos):

        cluster[idx] = C
        tipos[idx] = "core"

        i = 0
        while i < len(N):
            q = N[i]

            if q not in visitado:
                visitado.add(q)
                Nq = Algoritmo.vizinhanca_eps(D, q, eps)

                if len(Nq) >= min_pts:
                    tipos[q] = "core"
                    for x in Nq:
                        if x not in N:
                            N.append(x)
                else:
                    tipos[q] = "border"

            if q not in cluster:
                cluster[q] = C
                ruido.discard(q)

            i += 1