import os
import numpy as np
import argparse

from knn import *
from utils import *
from proposed import *


SETTINGS = ['passive', 'active']


def basic_accs(train_xs, train_ys, pred_ys, test_xs, test_ys):
    label_acc = sum(pred_ys == train_ys) / len(train_ys)
    knn_pred_ys = knn_run(train_xs, pred_ys, test_xs)
    knn_acc = sum(knn_pred_ys == test_ys) / len(test_ys)
    return label_acc, knn_acc


def single_proposed(xs, ys, ps, t, m, noise, ratio=0.8):
    indices = np.arange(len(ys))
    np.random.shuffle(indices)

    train_size = int(len(ys) * ratio)
    train_indices = indices[:train_size]
    train_xs, train_ys = xs[train_indices], ys[train_indices]
    train_ps = ps[train_indices]
    test_indices = indices[train_size:]
    test_xs, test_ys = xs[test_indices], ys[test_indices]
    pred_ys, _, _ = proposed_algo(train_ps, t, m, noise)
    return train_xs, train_ys, pred_ys, test_xs, test_ys, train_indices, test_indices


def single_coteaching_run(xs, hs, ys, ps, t, m, noise):
    train_hs, train_ys, pred_ys, test_hs, test_ys, train_indices, test_indices\
        = single_proposed(hs, ys, ps, t, m, noise)
    l_acc, k_acc = basic_accs(train_hs, train_ys, pred_ys, test_hs, test_ys)
    train_xs, test_xs = xs[train_indices], xs[test_indices]
    c_acc = coteaching(train_xs, pred_ys, test_xs, test_ys)
    return l_acc, k_acc, c_acc


def single_passive_run(xs, ys, ps, t, m, noise):
    train_xs, train_ys, pred_ys, test_xs, test_ys, _, _\
        = single_proposed(xs, ys, ps, t, m, noise)
    return basic_accs(train_xs, train_ys, pred_ys, test_xs, test_ys)


def passive_learning(dataset, run, t, m, noise):
    if 'cifar' in dataset:
        xs, hs, ys, ps = prepare_data(dataset)
        co_accs = []

        for _run in range(run):
            results = single_coteaching_run(xs, hs, ys, ps, t, m, noise)
            print('run {}/{}: label {:.4f}, knn {:.4f}, co {:.4f}'.format(
                _run + 1, run, results[0], results[1], results[2]))
    else:
        xs, ys, ps = prepare_data(dataset)

        for _run in range(run):
            results = single_passive_run(xs, ys, ps, t, m, noise)
            print('run {}/{}: label {:.4f}, knn {:.4f}'.format(
                _run + 1, run, results[0], results[1]))



def single_active_run(dataset, t, m, noise, eps, n):
    if dataset == 'gaussian':
        xs, ys, ps = prepare_gaussian(n, [[2, 2], [-2, -2]],
                                      [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        hs = np.arange(0, 1, 0.001)
        h_marks = np.array([True] * len(hs))

    hss, accs = [], []
    n_i = n // 10
    test_xs, test_ys = xs[-n_i:], ys[-n_i:]
    for i in range(int(np.ceil(np.log(1/eps)))):
        indices = range(i * n_i, (i + 1) * n_i)
        xs_i, ps_i, eps_i = xs[indices], ps[indices], 1 / (2 ** (i + 2))

        # preds: len(xs_i) * len(hs[h_marks])
        preds = np.tan(hs[h_marks][None, :] * np.pi) * xs_i[:, 0][:, None] \
                - xs_i[:, 1][:, None]
        preds = np.sum(preds > 0, axis=1)
        dis_marks = np.logical_and(0 < preds, preds < h_marks.sum())
        xs_i, ps_i = xs_i[dis_marks], ps_i[dis_marks]
        pred_ys_i, _, _ = proposed_algo(ps_i, t, m, noise)

        # preds: len(xs_i) * len(hs)
        preds = np.tan(hs[None, :] * np.pi) * xs_i[:, 0][:, None] \
                - xs_i[:, 1][:, None]
        preds = (np.sign(preds) + 1) / 2
        h_marks_i = (preds == pred_ys_i[:, None]).sum(axis=0) >= (1-eps_i) * sum(dis_marks)
        h_marks = np.logical_and(h_marks, h_marks_i)

        preds = np.tan(hs[h_marks][None, :] * np.pi) * test_xs[:, 0][:, None] \
                - test_xs[:, 1][:, None]
        preds = (np.sign(preds) + 1) / 2
        hss.append(sum(h_marks))
        accs.append((preds == test_ys[:, None]).sum() / (hss[-1] * n_i))
        print('Step {}: HS {}, ACC {:.4f}'.format(i + 1, hss[-1], accs[-1]))
    return hss, accs


def active_learning(dataset, run, t, m, noise, eps, n):
    for _run in range(run):
        print('run {}/{}:'.format(_run + 1, run))
        hss, accs = single_active_run(dataset, t, m, noise, eps, n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str,
                        choices=SETTINGS, default=SETTINGS[0])
    parser.add_argument('--dataset', type=str, default='mnist1v7')
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('-m', type=int, default=1)
    parser.add_argument('-t', type=int, default=20)
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('-n', type=int, default=10000)
    args = parser.parse_args()

    if args.setting == 'passive':
        passive_learning(args.dataset, args.run,
                         args.t, args.m, args.noise)
    else:
        active_learning(args.dataset, args.run,
                        args.t, args.m, args.noise, args.eps, args.n)
