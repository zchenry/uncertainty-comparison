import numpy as np

from usertopk import *
import torch
from torch import nn, from_numpy, optim
from os.path import exists
from knn import knn_run
from kmedoids import KMedoids


PATH = 'data/user'


def get_km(xs, n):
    km = KMedoids(n_cluster=n, max_iter=1000, tol=1e-5)
    km.fit(xs)
    kmidx = list(km.medoids)
    testidx = [i for i in range(len(xs)) if i not in kmidx]
    np.random.shuffle(testidx)
    return kmidx, testidx[:100]


def qsort(arr, res, cmps, n):
    if len(arr) <= 1:
        return arr, cmps
    else:
        for x in arr[1:]:
            cmps[arr[0], x] = 1
            cmps[x, arr[0]] = 1
        dices = np.random.binomial(1, res[arr[0], arr[1:]])
        lqs, lcmp = qsort([arr[1:][i] for i in range(len(arr[1:])) if dices[i] >= .5], res, cmps, n)
        rqs, rcmp = qsort([arr[1:][i] for i in range(len(arr[1:])) if dices[i] < .5], res, cmps, n)
        for i in range(n):
            for j in range(i, n):
                if lcmp[i, j] > 0 or rcmp[i, j] > 0:
                    cmps[i, j] = 1; cmps[j, i] = 1
        return lqs + [arr[0]] + rqs, cmps


def pos_ys(pos_res, abs_ys, params):
    n = len(abs_ys)
    cmps = np.zeros((n, n))

    indices, cmps = qsort(list(range(n)), pos_res, cmps, n)

    left, right, step = params
    while left < right - 1:
        mid = ((left + right) // 2) * step
        ids = mid + np.arange(step)
        #np.random.shuffle(ids)
        #k = max(step // 2, 1)
        #if abs_ys[ids[:k]].mean() > .5:
        if abs_ys[ids].mean() >= .5:
            right = mid // step
        else:
            left = mid // step
    if left == mid // step:
        mid += step
    _ys = np.zeros(n)
    _ys[indices[mid:]] = 1
    return _ys, cmps


def amb_ys(amb_res, pos_res, t):
    n = len(amb_res)
    indices = list(range(n))
    most_ambi_indices, cmpl, cmpr = topk(np.zeros(n), indices, t, amb_res)
    cmps = np.zeros((n, n))
    for i in most_ambi_indices:
        for j in range(n):
            cmps[i, j] = 1
            cmps[j, i] = 1
    _ys = (pos_res[most_ambi_indices].mean(axis=0) <= .5).astype(int)
    _ys[most_ambi_indices] = np.random.binomial(1, .5, t)
    return _ys, cmpl, cmpr, cmps


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, xs):
        return self.clf(xs)


def nn_run(xs, ys, test_xs, epoch):
    model = MLP().cuda()
    opt = optim.Adam(model.parameters(), amsgrad=True)

    xs = from_numpy(xs).float().cuda()
    ys = from_numpy(ys).float().cuda()[:, None]
    for e in range(epoch):
        model.train()
        opt.zero_grad()
        loss = nn.BCELoss()(model(xs), ys)
        loss.backward()
        opt.step()

    model.eval()
    test_xs = from_numpy(test_xs).float().cuda()
    _ys = model(test_xs).cpu().detach().numpy().reshape(-1)
    return (_ys >= .5).astype(int)


def accs(xs, ys, _ys, test_xs, test_ys, epoch):
    lacc = sum(ys == _ys) / len(ys)
    pred_ys = nn_run(xs, _ys, test_xs, epoch)
    tacc = sum(pred_ys == test_ys.astype(int)) / len(test_ys)
    knn_ys = knn_run(xs, _ys, test_xs)
    kacc = sum(knn_ys == test_ys.astype(int)) / len(test_ys)
    return lacc, tacc, kacc


def sim(abso):
    l = len(abso)
    n = 150
    posi = np.ones((l, n, n)) * .5
    ambi = np.ones((l, n, n)) * .5

    for u in range(l):
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = abso[u, i], abso[u, j]
                if pi > pj:
                    posi[u, i, j] = 0
                elif pi < pj:
                    posi[u, i, j] = 1
                posi[u, j, i] = 1 - posi[u, i, j]

                di, dj = np.abs(pi - 3), np.abs(pj - 3)
                if di < dj:
                    ambi[u, i, j] = 1
                elif di > dj:
                    ambi[u, i, j] = 0
                ambi[u, j, i] = 1 - ambi[u, i, j]
    return posi, ambi


def main():
    xs = np.load(f'{PATH}/car-simulation-xs.npz')['xs']
    data = np.load(f'{PATH}/car-simulation-annotation.npz')
    abso, absoraw = data['abso'], data['absoraw']
    posi, ambi = sim(absoraw)

    l = len(abso)
    for ni in range(10):
        n = ni * 5 + 5
        print(f'Using {n} images as training data')
        kmidx, testidx = get_km(xs, n)

        for u in range(l):
            print(f'User {u + 1}')
            epoch = 100

            _, tacc1, tacc2 = accs(xs[:n], abso[u, :n], abso[u, :n], xs[50:], abso[u, 50:], epoch)
            print(f'Uniform Selection, Full Supervision, MLP acc {tacc1 * 100:.2f}, KNN acc {tacc2 * 100:.2f}')

            _, tacc1, tacc2 = accs(xs[kmidx], abso[u, kmidx], abso[u, kmidx], xs[testidx], abso[u, testidx], epoch)
            print(f'Medoids, Full Supervision, MLP acc {tacc1 * 100:.2f}, KNN acc {tacc2 * 100:.2f}')

            params = [[0, 5, n // 5]]
            if n > 5:
                params.append([0, n // 5, 5])
                params.append([0, n // 4, 4])
                params.append([0, n // 3, 3])

            _bys, _best_acc = None, 0
            for p in params * 3:
                _ys, _ = pos_ys(posi[u, :n, :n], abso[u, :n], p)
                _acc = sum(_ys == abso[u, :n]) / len(_ys)
                if _acc > _best_acc or _bys is None:
                    _bys = _ys
                    _best_acc = _acc

            lacc, tacc1, tacc2 = accs(xs[:n], abso[u, :n], _bys, xs[50:], abso[u, 50:], epoch)
            print(f'Uniform Selection, Existing Method, label acc {lacc * 100:.2f}, MLP acc {tacc1 * 100:.2f}, KNN acc {tacc2 * 100:.2f}')

            _bys, _best_acc = None, 0
            for p in params * 3:
                _ys, _ = pos_ys(posi[u, kmidx][:, kmidx], abso[u, kmidx], p)
                _acc = sum(_ys == abso[u, kmidx]) / len(_ys)
                if _acc > _best_acc or _bys is None:
                    _bys = _ys
                    _best_acc = _acc
            lacc, tacc1, tacc2 = accs(xs[kmidx], abso[u, kmidx], _bys, xs[testidx], abso[u, testidx], epoch)
            print(f'Mediods, Existing Method, label acc {lacc * 100:.2f}, MLP acc {tacc1 * 100:.2f}, KNN acc {tacc2 * 100:.2f}')

            ts = [1, 3]
            if n > 5:
                ts = [1, 3, 5]
            if n > 15:
                ts = [1, 3, 5, 7]
            if n == 30:
                ts = [1, 3, 5]

            _bys, _best_acc = None, 0
            for t in ts * 3:
                _ys, _, _, _ = amb_ys(ambi[u, :n, :n], posi[u, :n, :n], t)
                _acc = sum(_ys == abso[u, :n]) / len(_ys)
                if _acc > _best_acc or _bys is None:
                    _bys = _ys
                    _best_acc = _acc
            lacc, tacc1, tacc2 = accs(xs[:n], abso[u, :n], _bys, xs[50:], abso[u, 50:], epoch)
            print(f'Uniform Selection, Proposed Method, label acc {lacc * 100:.2f}, MLP acc {tacc1 * 100:.2f}, KNN acc {tacc2 * 100:.2f}')

            _bys, _best_acc = None, 0
            for t in ts * 3:
                _ys, _, _, _ = amb_ys(ambi[u, kmidx][:, kmidx], posi[u, kmidx][:, kmidx], t)
                _acc = sum(_ys == abso[u, kmidx]) / len(_ys)
                if _acc > _best_acc or _bys is None:
                    _bys = _ys
                    _best_acc = _acc
            lacc, tacc1, tacc2 = accs(xs[kmidx], abso[u, kmidx], _bys, xs[testidx], abso[u, testidx], t)
            print(f'Mediods, Proposed Method, label acc {lacc * 100:.2f}, MLP acc {tacc1 * 100:.2f}, KNN acc {tacc2 * 100:.2f}')


if __name__ == '__main__':
    main()
