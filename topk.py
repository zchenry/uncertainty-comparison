import numpy as np


def compare(xs, i, j, m, acc):
    result = xs[i] < xs[j]
    runs = sum(np.random.binomial(m, acc, 1))
    if runs > m / 2:
        return result
    else:
        return not result


def select(xs, indices, m, acc, cmpl, cmpr):
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
            if compare(xs, compare_i, compare_j, m, acc):
                won_indices[r] = won_indices[2*r]
            else:
                won_indices[r] = won_indices[2*r+1]
        layer_n = rounds
    return indices[won_indices[0]], cmpl, cmpr


def heapify(indices, xs, i, m, acc, cmpl, cmpr):
    left = 2 * i + 1
    right = left + 1

    if left + 1 <= len(indices):
        cmpl.append(indices[left]); cmpr.append(indices[i])
    if left + 1 <= len(indices) and compare(xs, indices[left], indices[i], m, acc):
        min_i = left
    else:
        min_i = i

    if right + 1 <= len(indices):
        cmpl.append(indices[right]); cmpr.append(indices[min_i])
    if right + 1 <= len(indices) and compare(xs, indices[right], indices[min_i], m, acc):
        min_i = right

    if min_i != i:
        tmp = indices[i]; indices[i] = indices[min_i]; indices[min_i] = tmp
        indices, cmpl, cmpr = heapify(indices, xs, min_i, m, acc, cmpl, cmpr)

    return indices, cmpl, cmpr


def build_heap(indices, xs, m, acc, cmpl, cmpr):
    start_index = int(np.floor(len(indices) / 2)) - 1
    for i in range(start_index, -1, -1):
        indices, cmpl, cmpr = heapify(indices, xs, i, m, acc, cmpl, cmpr)
    return indices, cmpl, cmpr


def topk(xs, indices, k, m, acc):
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
        i_top_index, cmpl, cmpr = select(xs, i_indices, m, acc, cmpl, cmpr)

        top_indices.append(i_top_index)
        i_indices.remove(i_top_index)
        indices_arrays.append(i_indices)
    top_indices, cmpl, cmpr = build_heap(top_indices, xs, m, acc, cmpl, cmpr)

    for i in range(k - 1):
        top_k.append(top_indices[0])
        j = int(np.floor(top_k[-1] / q))
        if len(indices_arrays[j]) > 0:
            j_top_index, cmpl, cmpr = select(
                xs, indices_arrays[j], m, acc, cmpl, cmpr)
            indices_arrays[j].remove(j_top_index)
            top_indices[0] = j_top_index
            top_indices, cmpl, cmpr = heapify(
                top_indices, xs, 0, m, acc, cmpl, cmpr)
    top_k.append(top_indices[0])

    return top_k, cmpl, cmpr


if __name__ == '__main__':
    n = 10000
    k = 4
    m = 10
    acc = 0.8

    for _ in range(10):
        xs, indices = list(range(n)), list(range(n))
        np.random.shuffle(xs)
        results, _, _ = topk(xs, indices, k, m, acc)
        print([xs[i] for i in results])
