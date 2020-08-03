import numpy as np


def compare(xs, i, j, res):
    return res[i, j] == 1


def select(xs, indices, res, cmpl, cmpr):
    n = len(indices)
    won_indices = np.arange(n)
    layer_n = n

    for layer in range(int(np.ceil(np.log2(n)))):
        rounds = int(np.ceil(layer_n / 2))
        for r in range(rounds):
            if (r+1)*2 > layer_n:
                won_indices[r] = won_indices[2*r]
                continue

            compare_i = indices[won_indices[2*r]]
            compare_j = indices[won_indices[2*r+1]]
            cmpl.append(compare_i); cmpr.append(compare_j)
            if compare(xs, compare_i, compare_j, res):
                won_indices[r] = won_indices[2*r]
            else:
                won_indices[r] = won_indices[2*r+1]
        layer_n = rounds
    return indices[won_indices[0]], cmpl, cmpr


def heapify(indices, xs, i, res, cmpl, cmpr):
    left = 2 * i + 1
    right = left + 1

    if left + 1 <= len(indices):
        cmpl.append(indices[left]); cmpr.append(indices[i])
    if left + 1 <= len(indices) and compare(xs, indices[left], indices[i], res):
        min_i = left
    else:
        min_i = i

    if right + 1 <= len(indices):
        cmpl.append(indices[right]); cmpr.append(indices[min_i])
    if right + 1 <= len(indices) and compare(xs, indices[right], indices[min_i], res):
        min_i = right

    if min_i != i:
        tmp = indices[i]; indices[i] = indices[min_i]; indices[min_i] = tmp
        indices, cmpl, cmpr = heapify(indices, xs, min_i, res, cmpl, cmpr)

    return indices, cmpl, cmpr


def build_heap(indices, xs, res, cmpl, cmpr):
    start_index = int(np.floor(len(indices) / 2)) - 1
    for i in range(start_index, -1, -1):
        indices, cmpl, cmpr = heapify(indices, xs, i, res, cmpl, cmpr)
    return indices, cmpl, cmpr


def topk(xs, indices, k, res):
    n = len(indices)
    q = int(np.ceil(n / k))
    if q*(k-1) >= n:
        q -= 1
    indices_arrays, top_indices, top_k = [], [], []
    cmpl, cmpr = [], []

    for i in range(k):
        start_index = i * q
        end_index = start_index + q if i + 1 < k else n
        i_indices = list(range(start_index, end_index))
        i_top_index, cmpl, cmpr = select(xs, i_indices, res, cmpl, cmpr)

        top_indices.append(i_top_index)
        i_indices.remove(i_top_index)
        indices_arrays.append(i_indices)
    top_indices, cmpl, cmpr = build_heap(top_indices, xs, res, cmpl, cmpr)

    for i in range(k - 1):
        top_k.append(top_indices[0])
        j = int(np.floor(top_k[-1] / q))
        if len(indices_arrays[j]) > 0:
            j_top_index, cmpl, cmpr = select(
                xs, indices_arrays[j], res, cmpl, cmpr)
            indices_arrays[j].remove(j_top_index)
            top_indices[0] = j_top_index
            top_indices, cmpl, cmpr = heapify(
                top_indices, xs, 0, res, cmpl, cmpr)
    top_k.append(top_indices[0])

    return top_k, cmpl, cmpr
