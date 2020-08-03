import argparse
import numpy as np

from collections import OrderedDict
from usertopk import *
import torch
from torch import nn, from_numpy, optim
from os.path import exists
import torch.utils.model_zoo as model_zoo


N = 25
U = 10
TEST_N = 100
PATH = 'data/user'


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)
        self.model = nn.Sequential(layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.model.forward(input)


def mlp_run(xs, ys, test_xs, epochs):
    model = MLP(784, [256, 256], 10).cuda()
    m = model_zoo.load_url(
        'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth',
        model_dir=PATH)
    state_dict = m.state_dict() if isinstance(m, nn.Module) else m
    model.load_state_dict(state_dict)

    opt = optim.AdamW(model.parameters(), amsgrad=True)
    for layer in ['fc1', 'drop1']:
        m = getattr(model.model, layer)
        if hasattr(m, 'weight'):
            m.weight.requires_grad = False
            m.weight.grad = None
        if hasattr(m, 'bias'):
            m.bias.requires_grad = False
            m.bias.grad = None

    xs = from_numpy(xs).float().cuda() / 255
    ys = from_numpy(ys).float().cuda()[:, None]

    for epoch in range(epochs):
        opt.zero_grad()
        nn.BCELoss()(nn.Sigmoid()(model(xs)[:, [0]]), ys).backward()
        opt.step()

    model.eval()
    _ys = model(from_numpy(test_xs).float().cuda())[:, 0].cpu().detach().numpy()
    return (_ys >= 0).astype(int)


def accs(xs, ys, _ys, test_xs, test_ys, epochs):
    ys_acc = sum(ys == _ys) / len(ys)
    _pred_ys = mlp_run(xs, _ys, test_xs, epochs)
    pred_ys_acc = sum(test_ys == _pred_ys) / TEST_N
    return ys_acc, pred_ys_acc


def qsort(arr, res, cmps, n):
    if len(arr) <= 1:
        return arr, cmps
    else:
        for x in arr[1:]:
            cmps[arr[0], x] = 1
            cmps[x, arr[0]] = 1
        lqs, lcmp = qsort([x for x in arr[1:] if res[x, arr[0]] < .5], res, cmps, n)
        rqs, rcmp = qsort([x for x in arr[1:] if res[x, arr[0]] >= .5], res, cmps, n)
        for i in range(n):
            for j in range(i, n):
                if lcmp[i, j] > 0 or rcmp[i, j] > 0:
                    cmps[i, j] = 1; cmps[j, i] = 1
        return lqs + [arr[0]] + rqs, cmps


def pos_ys(pos_res, abs_ys):
    n = len(abs_ys)
    cmps = np.zeros((n, n))

    indices, cmps = qsort(list(range(n)), pos_res, cmps, n)
    left, right, step = 0, 5, n // 5

    while left < right - 1:
        mid = ((left + right) // 2) * step
        ids = mid + np.arange(step)
        np.random.shuffle(ids)
        k = max(step // 2, 1)
        if abs_ys[ids[:k]].mean() > .5:
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


def main(setting):
    test_data = np.load(f'{PATH}/kuzushiji-{setting}-test.npz')
    test_xs, test_ys = test_data['xs'], test_data['ys']
    test_xs = test_xs.reshape(len(test_ys), -1)
    ids = np.arange(len(test_ys))
    np.random.shuffle(ids)
    test_xs, test_ys = test_xs[ids[:TEST_N]], test_ys[ids[:TEST_N]]

    data = np.load(f'{PATH}/kuzushiji-{setting}-train.npz')
    xs, ys = data['xs'], data['ys']

    data = np.load(f'{PATH}/kuzushiji-{setting}-annotation.npz')
    abs_ys, abs_diff = data['abs_ys'], data['abs_diff']
    pos_res, pos_diff = data['pos_res'], data['pos_diff']
    amb_res, amb_diff = data['amb_res'], data['amb_diff']

    ts = [1, 1, 3, 3, 3]
    epochs = [100, 100, 100, 100, 100]

    for ni in range(5):
        n = [5, 10, 15, 20, 25][ni]
        print(n)

        print('Full Supervision')
        for u in range(U):
            _ys_acc, _pred_ys_acc = accs(
                xs[:n], ys[:n], abs_ys[:n, u], test_xs, test_ys, epochs[ni])
            print(f'User {u + 1}, label acc {_ys_acc*100:.2f}, test acc {_pred_ys_acc*100:.2f}')
        abs_ys_m = (abs_ys[:n].mean(axis=1) >= .5).astype(int)
        _ys_acc, _pred_ys_acc = accs(
            xs[:n], ys[:n], abs_ys_m, test_xs, test_ys, epochs[ni])
        print(f'Aggregated, label acc {_ys_acc*100:.2f}, test acc {_pred_ys_acc*100:.2f}')

        print('Positivity Comparison + Explicit Labeling')
        for u in range(U):
            _ys, _ = pos_ys(pos_res[:n, :n, u], abs_ys[:n, u])
            _ys_acc, _pred_ys_acc = accs(
                xs[:n], ys[:n], _ys[:n], test_xs, test_ys, epochs[ni])
            print(f'User {u + 1}, label acc {_ys_acc*100:.2f}, test acc {_pred_ys_acc*100:.2f}')
        pos_res_m = (pos_res[:n, :n].mean(axis=2) > .5).astype(int)
        _ys, _ = pos_ys(pos_res_m, abs_ys_m)
        _ys_acc, _pred_ys_acc = accs(
            xs[:n], ys[:n], _ys[:n], test_xs, test_ys, epochs[ni])
        print(f'Aggregated, label acc {_ys_acc*100:.2f}, test acc {_pred_ys_acc*100:.2f}')

        print('Proposed Method: Ambiguity Comparison + Positivity Comparison')
        for u in range(U):
            _ys, cmpl, cmpr, cmps = amb_ys(
                amb_res[:n, :n, u], pos_res[:n, :n, u], ts[ni])
            _ys_acc, _pred_ys_acc = accs(
                xs[:n], ys[:n], _ys[:n], test_xs, test_ys, epochs[ni])
            print(f'User {u + 1}, label acc {_ys_acc*100:.2f}, test acc {_pred_ys_acc*100:.2f}')
        amb_res_m = (amb_res[:n, :n].mean(axis=2) > .5).astype(int)
        _ys, _, _, _ = amb_ys(amb_res_m, pos_res_m, ts[ni])
        _ys_acc, _pred_ys_acc = accs(
            xs[:n], ys[:n], _ys[:n], test_xs, test_ys, epochs[ni])
        print(f'Aggregated, label acc {_ys_acc*100:.2f}, test acc {_pred_ys_acc*100:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str,
                        choices=['medoids', 'uniform'],
                        default='medoids')
    args = parser.parse_args()
    main(args.setting)
