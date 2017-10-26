from math import sqrt

import numpy
import numpy.linalg as lin
import time
import os
from rtree import index


def process_input():
    mat = numpy.fromfile("ColorHistogram.asc", dtype=numpy.float64, sep="\n")
    l = int(mat.shape[0] / 33)
    return mat.reshape((l, 33))[:, 1:]


def query_multi(records, mat, v, avg, tree, qux_cor, k):
    br = numpy.dot(qux_cor - avg, v[:, :k])
    D_NN = numpy.inf
    NN = -1
    nearest_list = tree.nearest(numpy.concatenate([br, br]), num_results=len(records))

    # count = 0
    for near in nearest_list:
        d = sum((mat[near] - br)**2)
        # count += 1
        if d < D_NN:
            D = sum((records[near] - qux_cor)**2)
            if D < D_NN:
                D_NN = D
                NN = near
        else:
            # print("find it " + str(count))
            break
    # print(str(NN) + " " + str(D_NN))
    return NN, D_NN


def query_two(records, v, avg, tree, qux_cor, k):
    br = numpy.dot(qux_cor - avg, v[:, :k])
    nearest_list = list(tree.nearest(numpy.concatenate([br, br])))
    DD = sum((records[nearest_list[0]] - qux_cor) ** 2)
    DD_x = nearest_list[0]
    for nearest in nearest_list:
        DD_tmp = sum((records[nearest] - qux_cor) ** 2)
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
    return d_x, min_d


def query_linear(records, qux_cor):
    min_d = sum((records[0] - qux_cor)**2)
    min_dx = 0
    for idx, record in enumerate(records):
        d = sum((record - qux_cor)**2)
        if d < min_d:
            min_d = d
            min_dx = idx
    return min_dx, min_d


if __name__ == "__main__":

    if os.path.exists("3d_index.dat"):
        os.remove("3d_index.dat")
        os.remove("3d_index.idx")

    k = 28
    n = 100
    mat = process_input()
    min_mat = numpy.min(mat, 0)
    max_mat = numpy.max(mat, 0)

    records = mat.tolist()

    start = time.time()
    avg = numpy.mean(mat, axis=0)
    mat = mat - avg

    # u, s, vt = lin.svd(mat, full_matrices=False)
    # v = vt.T
    # S = numpy.diag(s)
    # B = numpy.dot(mat, v[:, :k])

    e, v = lin.eig(numpy.dot(mat.T, mat) / (mat.shape[0] - 1))
    idx = e.argsort()[::-1]
    e, v = e[idx], v[idx]
    B = numpy.dot(mat, v[:, :k])

    p = index.Property()
    p.dimension = k
    idxkd = index.Index('3d_index', properties=p)
    for idx, r in enumerate(B):
        idxkd.insert(idx, numpy.concatenate([r, r]), r)
    print("time for building index: " + str(time.time() - start) + "s")


    print("start generating dataset")
    datasets = []
    for i in range(n):
        r = numpy.random.rand(1, 32)[0]
        rec = min_mat + r * (max_mat - min_mat) * r
        datasets.append(rec)

    print("start queries")

    # for idx, b_r in enumerate(datasets):
    #     r_s = query_two(records, v.T, avg, idxkd, b_r, k)
    #     r_l = query_linear(records, b_r)
    #     r_m = query_multi(records, B, v, avg, idxkd, b_r, k)
    #     print(r_s)
    #     print(r_l)
    #     print(r_m)
    #     print("")

    start = time.time()
    for idx, b_r in enumerate(datasets):
        r_s = query_two(records, v, avg, idxkd, b_r, k)
    print("time for two step: " + str(time.time() - start) + "s")

    start = time.time()
    for idx, b_r in enumerate(datasets):
        r_l = query_multi(records, B, v, avg, idxkd, b_r, k)
    print("time for multi-step: " + str(time.time() - start) + "s")

    start = time.time()
    for idx, b_r in enumerate(datasets):
        r_l = query_linear(records, b_r)
    print("time for linear search: " + str(time.time() - start) + "s")
