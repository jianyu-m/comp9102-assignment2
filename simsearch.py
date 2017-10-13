import numpy
import numpy.linalg as lin
from rtree import index


def process_input():
    mat = numpy.fromfile("ColorHistogram.asc", dtype=numpy.float64, sep="\n")
    l = int(mat.shape[0] / 33)
    return mat.reshape((l, 33))[:100, 1:]

if __name__ == "__main__":
    k = 10
    mat = process_input()
    avg = numpy.mean(mat, axis=0)
    mat = mat - avg
    u, s, v = lin.svd(mat)
    S = numpy.diag(s)
    B = numpy.dot(u[:, :k], S[:k, :k])

    p = index.Property()
    p.dimension = k
    idxkd = index.Index('3d_index', properties=p)
    for idx, r in enumerate(B):
        idxkd.insert(idx + 1, numpy.concatenate([r, r]), r)

    br = numpy.dot(mat[1], numpy.transpose(v))[:k]
    for i in idxkd.nearest(numpy.concatenate([br, br])):
        print(i)
