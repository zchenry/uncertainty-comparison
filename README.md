This repository contains codes and collected user study data for the paper [Classification from Ambiguity Comparisons](http://arxiv.org/abs/2008.00645).

`Python 3.6.2` is used. Dependencies are specified in the `requirements.txt` file.

### Simulation Study
Options of the execution file `simulation_study.py` are
- `--setting`: Setting `passive` for the sufficient budget case and `active` for the insufficient budget case.
- `--dataset`: Selections for sufficient budget are `mnist1v7`, `mnist3v5`, `fashion0v6`, `fashion2v4`, `kuzushi1v7`, `kuzushi2v6`, `cifar1v9` and `cifar4v7`. Dataset for insufficient budget is `gaussian`.
- `--run`: Times for independent repetitions.
- `-m`: The hyper-parameter for repetition number of a single query.
- `-t`: The hyper-parameter for the size of the delegation subset.
- `--noise`: The noise rate for both oracles.
- `--eps`: The precision hyper-parameter for active learning.
- `-n`: Number of generated data points for active learning.

#### Sample Result
Command `python simulation_study.py --setting passive --dataset mnist1v7 --run 1 -m 5 -t 20 --noise 0.3` will generate the following results.
```
run 1/10: label 0.9665, knn 0.9881
```

Command `python simulation_study.py --setting passive --dataset cifar1v9 --run 1 -m 5 -t 20 --noise 0.3` will finally generate the following results.
```
run 1/1: label 0.9655, knn 0.9980, co 0.8440
```

Command `python simulation_study.py --setting active --dataset gaussian --run 1 --noise 0.3 --eps 0.1 -n 10000` will generate the following results.
```
Step 1: HS 655, ACC 0.9767
Step 2: HS 460, ACC 0.9950
Step 3: HS 31, ACC 0.9990
```

### User Study

#### User Study on Difficulty of selected Kuzushiji Images pairs
The data files are `data/user/collect25_1.csv` and `data/user/collect25_2.csv`.

Command `python user_study_kuzushiji_difficulty.py` will generate the following results.
```
mean 2.7486, std 0.3421
Ttest_1sampResult(statistic=-5.145177699062356, pvalue=4.693217551403353e-06)
```

#### User Study on Kuzushiji Images using simulated pairwise comparisons
The data file is `data/user/kuzushiji-medoids-simulation.csv`.

It has `20` rows, with each row indicating the results from one user.

It has `100` columns, as the explicit labeling and its difficulty are collected for `50` medoids.

#### User Study on Kuzushiji Images using actual pairwise annotations
The data file is `data/user/kuzushiji-medoids-annotation.npz` and `data/user/kuzushiji-uniform-annotation.npz`.

Inside the files, the array `'pos_res'` has size `(25*25*10)` and containes the pairwise annotation for all possible pairs among `25` selected images from `10` users.
Similarly, the array `'amb_res'` in the files has the same size.

The file `user_study_kuzushiji_feedback.py` has only one option `--setting`.
This option can be set as `medoids` or `uniform`.

Command `python user_study_kuzushiji_feedback.py --setting medoids` will evaluate methods on the medoids feedback.

#### User Study on Car Images using simulated pairwise comparisons
The data file is `data/user/car-simulation-annotation.npz`.

In the file, the array `'abso'` has shape `(4*150)` as four users are queried on `150` images.

In `user_study_car.py`, the function `sim()` constructs pairwise comparison results from the `'abso'` array.

Command `python user_study_car.py` will evalute methods on the data.

### Citation
If you use the proposed method or any of the data in your work, we would appreciate a reference to our paper:
```
@online{cui2020,
  author       = {Zhenghang Cui and Issei Sato},
  title        = {Classification from Ambiguity Comparisons},
  year         = {2020},
  eprinttype   = {arXiv:2008.00645},
}
```
