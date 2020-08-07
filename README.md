# A deep-learning-based method for detecting workarounds in event log data

## Steps of the method
1. Split event log in data set and prediction set (90%/10%)
2. Filter out noise from data set by using an Autoencoder model
3. Integration of workarounds (rules enriched by random elements)
4. Learn a CNN model for detecting workarounds 
    - based on 80% of the data set
    - test model on 20% of the data set 
    - hyper-parameter optimization with a 10%-validation-set of the training set
5. Apply the CNN model on the prediction set including noise (10%)

## Setting
1. Binary encoded categorical attributes (less sparse than one-hot encoding)
2. An Autoencoder; no hyper-parameter optimization; threshold not optimized 
3. -
4. Split validation + hyper-parameter optimisation (TPE)
5. -

## Paper
This repository contains the implementation of the work from XY.
A preliminary version of this method was presented in:
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






