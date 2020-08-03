from topk import *


def proposed_algo(ps, k, m, noise):
    n = len(ps)
    indices = list(range(n))
    ambi_ps = np.abs(ps - 0.5)

    most_ambi_indices, cmpl, cmpr = topk(ambi_ps, indices, k, m, 1 - noise)

    true_results = ps[:, None] > ps[most_ambi_indices][None, :]
    randomness = np.random.binomial(1, 1 - noise, true_results.shape) == 1
    results = np.logical_not(np.logical_xor(true_results, randomness))
    pred_ys = (results.sum(axis=1) > (k/2)).astype(int)
    return pred_ys, cmpl, cmpr


if __name__ == '__main__':
    m = 20
    noise = 0.2

    for _ in range(1):
        ps = np.arange(0, 1, 1e-4)
        np.random.shuffle(ps)
        ys = [1 if p > 0.5 else 0 for p in ps]
        n = len(ps)

        pred_ys, _, _ = proposed_algo(ps, 50, m, noise)
        print(sum(pred_ys == ys) / n)
