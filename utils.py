import os
from os.path import exists
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, Module, init, Conv2d, Linear, BatchNorm2d
from torch import save, load, flatten, cat, FloatTensor, LongTensor, max, is_tensor
from torch import sum as tsum
from torch.autograd import Variable


def printr(s):
    print("\r{}".format(" " * 50), end="")
    print("\r{}".format(s), end="")


def fetch(dataset):
    return fetch_openml(dataset, return_X_y=True, data_home='data')


def prepare_gaussian(n, means, stds):
    positive_xs = np.random.multivariate_normal(means[0], stds[0], n)
    negative_xs = np.random.multivariate_normal(means[1], stds[1], n)

    ys = np.random.binomial(1, 0.5, n)
    xs = np.concatenate((positive_xs[:, None], negative_xs[:, None]), axis=1)
    xs = xs[tuple([list(range(n)), ys])]

    clf = LogisticRegression(n_jobs=-1, max_iter=100000)
    clf.fit(xs, ys)
    ps = clf.predict_proba(xs)[:, 1]
    return xs, ys, ps


def prepare_data(dataset):
    data_path = 'data/{}'.format(dataset)
    data_full_path = '{}.npz'.format(data_path)
    if exists(data_full_path):
        data = np.load(data_full_path, allow_pickle=True)
        if 'cifar' in dataset or 'celeb' in dataset:
            return data['xs'], data['hs'], data['ys'], data['ps']
        else:
            return data['xs'], data['ys'], data['ps']

    if 'cifar' in dataset:
        device = 'cuda:0'
        model = models.resnet152(num_classes=10).to(device)
        model_path = 'data/cifar_resnet152'
        transform = Compose(
            [ToTensor(),
             Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        data = CIFAR10(root='data', download=True, transform=transform)

        if exists(model_path):
            model.load_state_dict(load(model_path))
        else:
            loader = DataLoader(data, batch_size=1024, shuffle=True)
            loss = CrossEntropyLoss()
            optim = AdamW(model.parameters(), amsgrad=True)

            for epoch in range(100):
                epoch_loss = 0
                for _data in loader:
                    xs, ys = _data
                    optim.zero_grad()
                    _loss = loss(model(xs.to(device)), ys.to(device))
                    _loss.backward()
                    optim.step()

                    epoch_loss += _loss.item()
                print('Epoch {}: Loss: {}'.format(epoch + 1, epoch_loss))
            save(model.state_dict(), model_path)

        xs, ys = data.data, np.array(data.targets)
        first_digit, second_digit = int(dataset[5]), int(dataset[7])
        mask = np.logical_or(ys == first_digit, ys == second_digit)
        xs = xs[mask]
        ys = np.array([1 if y == first_digit else 0 for y in ys[mask]])

        loader = DataLoader(data, batch_size=1024)
        hs = None
        model.eval()
        for _data in loader:
            _xs, _ys = _data
            _mask = (_ys == first_digit) | (_ys == second_digit)
            if len(_mask) > 0:
                _hs = resnet_fc(model, _xs[_mask].to(device))
                _hs = _hs.cpu().detach().numpy()
                hs = _hs if hs is None else np.concatenate((hs, _hs), axis=0)

        clf = LogisticRegression(n_jobs=-1, max_iter=100000)
        clf.fit(hs, ys)
        ps = clf.predict_proba(hs)[:, 1]

        np.savez(data_path, xs=xs, hs=hs, ys=ys, ps=ps)
        return xs, hs, ys, ps

    if 'mnist' in dataset:
        xs, ys = fetch('mnist_784')
        first_digit, second_digit = dataset[5], dataset[7]
    elif 'fashion' in dataset:
        xs, ys = fetch('Fashion-MNIST')
        first_digit, second_digit = dataset[7], dataset[9]
    elif 'kuzushi' in dataset:
        xs, ys = fetch('Kuzushiji-MNIST')
        first_digit, second_digit = dataset[7], dataset[9]
    mask = np.logical_or(ys == first_digit, ys == second_digit)
    xs = xs[mask] / 255
    ys = np.array([1 if y == first_digit else 0 for y in ys[mask]])

    clf = LogisticRegression(n_jobs=-1, max_iter=100000)
    clf.fit(xs, ys)
    ps = clf.predict_proba(xs)[:, 1]

    np.savez(data_path, xs=xs, ys=ys, ps=ps)
    return xs, ys, ps

def resnet_fc(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return flatten(model.avgpool(x), 1)


def accuracy(logit, target, batch_size, topk=1):
    output = F.softmax(logit, dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
    return float(correct_k.mul_(100.0 / batch_size))


def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return tsum(loss_1_update)/num_remember, tsum(loss_2_update)/num_remember


def coteaching(train_xs, train_ys, test_xs, test_ys):
    train_xs = np.moveaxis(train_xs, 3, 1)
    test_xs = np.moveaxis(test_xs, 3, 1)

    batch_size = 1024
    train_loader = DataLoader(
        TensorDataset(FloatTensor(train_xs), LongTensor(train_ys)),
        batch_size=batch_size)

    n_epoch, forget_rate = 100, 0.2
    rate_schedule = np.ones(n_epoch) * forget_rate
    rate_schedule[0] = 0.0

    device = 'cuda:0'
    model1 = models.resnet18(num_classes=2).to(device)
    optim1 = AdamW(model1.parameters())
    model2 = models.resnet18(num_classes=2).to(device)
    optim2 = AdamW(model2.parameters())

    for epoch in range(1, n_epoch):
        iters, acc1, acc2= 0, 0, 0
        for (images, labels) in train_loader:
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            iters += 1
            logits1 = model1(images)
            acc1 += accuracy(logits1, labels, batch_size)
            logits2 = model2(images)
            acc2 += accuracy(logits2, labels, batch_size)

            loss_1, loss_2 = loss_coteaching(logits1, logits2, labels,
                                             rate_schedule[epoch])
            optim1.zero_grad()
            loss_1.backward()
            optim1.step()
            optim2.zero_grad()
            loss_2.backward()
            optim2.step()

        printr('Coteaching: Epoch {}: acc1 {:.4f}, acc2 {:.4f}'.format(
            epoch, acc1 / iters, acc2 / iters))
    printr('')
    test_loader = DataLoader(
        TensorDataset(FloatTensor(test_xs), LongTensor(test_ys)),
        batch_size=1024)

    def _eval(model):
        total, correct = 0, 0
        model.eval()
        for images, labels in test_loader:
            _, preds = max(F.softmax(model(images.cuda()), dim=1).data, 1)
            total += len(labels)
            correct += int((preds.cpu() == labels).sum())
        return correct / total

    return (_eval(model1) + _eval(model2)) / 2
