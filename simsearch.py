from math import sqrt

import numpy
import numpy.linalg as lin
import time
from matplotlib.mlab import PCA
from rtree import index


def process_input():
    mat = numpy.fromfile("ColorHistogram.asc", dtype=numpy.float64, sep="\n")
    l = int(mat.shape[0] / 33)
    return mat.reshape((l, 33))[:, 1:]


def query_multi(records, v, avg, tree, qux_cor, k):
    br = numpy.dot(qux_cor - avg, numpy.transpose(v))[:k]
    D = numpy.inf


def query_two(records, v, avg, tree, qux_cor, k):
    br = numpy.dot(qux_cor - avg, numpy.transpose(v))[:k]
    nearest_list = list(tree.nearest(numpy.concatenate([br, br])))
    DD = sum((records[nearest_list[0]] - qux_cor) ** 2)
    DD_x = nearest_list[0]
    count = 0
    for nearest in nearest_list:
        DD_tmp = sum((records[nearest] - qux_cor) ** 2)
        count += 1
        if DD_tmp < DD:
            DD = DD_tmp
            DD_x = nearest
    d_x = DD_x

    D = sqrt(DD)
    min_d = DD
    touch_records = idxkd.intersection(numpy.concatenate([br - D, br + D]))
    for idx in touch_records:
        d = sum((records[idx] - qux_cor) ** 2)
        if d < min_d:
            min_d = d
            d_x = idx
    return d_x


def query_linear(records, qux_cor):
    min_d = sum((records[0] - qux_cor)**2)
    min_dx = 0
    for idx, record in enumerate(records):
        d = sum((record - qux_cor)**2)
        if d < min_d:
            min_d = d
            min_dx = idx
    return min_dx


if __name__ == "__main__":
    k = 10
    mat = process_input()
    min_mat = numpy.min(mat, 0)
    max_mat = numpy.max(mat, 0)

    records = mat.tolist()

    avg = numpy.mean(mat, axis=0)
    mat = mat - avg
    u, s, v = lin.svd(mat, full_matrices=False)
    S = numpy.diag(s)
    B = numpy.dot(u[:, :k], S[:k, :k])

    p = index.Property()
    p.dimension = k
    idxkd = index.Index('3d_index', properties=p)
    for idx, r in enumerate(B):
        idxkd.insert(idx, numpy.concatenate([r, r]), r)

    print("start generating dataset")
    datasets = []
    for i in range(10):
        r = numpy.random.rand(1, 32)[0]
        rec = min_mat + r * (max_mat - min_mat) * r
        datasets.append(rec)

    print("start queries")
    start = time.time()
    for idx, b_r in enumerate(datasets):
        r_s = query_two(records, v, avg, idxkd, b_r, k)
    print(time.time() - start)

    start = time.time()
    for idx, b_r in enumerate(datasets):
        r_l = query_linear(records, b_r)
    print(time.time() - start)
