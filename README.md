# A deep-learning-based method for detecting workarounds in event log data

## How to use the method?
...

## Steps of the method
1. Split event log in data set and prediction set (90%/10%)
2. Filter out noise from data set by using an Autoencoder (AE) model
3. Integration of workarounds (rules enriched by random elements, 30% of the instances include a workaround)
4. Learn a CNN model for detecting workarounds 
    - based on 80% of the data set
    - test model on 20% of the data set 
    - hyper-parameter optimization with a 10%-validation-set of the training set
5. Apply the CNN model on the prediction set including noise (10%)

## Setting
- Metrics: 
    - accuracy (macro), precision (macro) and recall (macro) -> to not ignore workaround classes with fewer instances  
    - auc_roc (macro, according to Fawcett (2006))
- Encoding: binary encoded categorical attributes (less sparse than one-hot encoding)
- HPO
    - Not for AE + threshold not optimized 
    - For CNN via TPE
- Shuffling:
    - For data set splits: yes
    - For instances in epochs: no
- Seed: no

## Further details
- For both models (i.e. AE and CNN), we do not shuffle the instances per epoch.
- Ensuring reproducible results via a seed (only for testing the implementation):
    - np.random.seed(1)
    - tf.random.set_seed(1)
    - random.seed(1)
    - optuna.samplers.TPESampler(1)


## Paper
This repository contains the implementation of [Weinzierl et al. (2020)](https://arxiv.org/). A previous version of this method was presented in:


```
@proceedings{weinzierl2020wdwdl,
    title={Detecting Workarounds in Business Processes -- a Deep Learning method for Analyzing Event Logs},
    author={Sven Weinzierl and Verena Wolf and Tobias Pauli and Daniel Beverungen and Martin Matzner},
    year={2020},
    booktitle={Proceedings of the 28th European Conference on Information Systems (ECIS2020)},
    year={2020},
    pages={1-16}
    organization={AISeL}
}
```






