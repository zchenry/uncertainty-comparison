import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp as test


PATH = 'data/user'


def read_csv(fname, c):
    return pd.read_csv('{}/{}.csv'.format(PATH, fname),
                       usecols=np.arange(3, c + 3))


def main():
    v1 = read_csv(f'collect25_1', 21 * 4).values[:, 3::4]
    v2 = read_csv(f'collect25_2', 21 * 4).values[:, 3::4]
    vs = np.concatenate((v1, v2), axis=1)

    vs = vs.mean(axis=1)
    print(f'mean {vs.mean():.4f}, std {vs.std():.4f}')
    print(test(vs, 3))


if __name__ == '__main__':
    main()
