# A deep-learning-based method for detecting workarounds in event log data

## How to use the method?
1. Create a virual environment (you find further details about that [here](https://docs.python.org/3/tutorial/venv.html))
2. Install the required python packages through the command "pip install -r requirements.txt" (requirements.txt)
3. Set relevant parameters in the file "config.py" (wdwdl/config.py)
4. Execute the file "main.py" (wdwdl/main.py)

## Steps of the method
1. Split event log in data set and prediction set (90%/10%)
2. Filter out noise from data set by using an Autoencoder (AE) model
3. Integration of workarounds (rules enriched by random elements, 30% of the instances include a workaround)
4. Learn a convolutional neural network (CNN) model for detecting workarounds 
    - Based on 80% of the data set
    - Test model on 20% of the data set 
    - Hyper-parameter optimization (HPO) with a 10%-validation-set of the training set
5. Apply the CNN model on the prediction set (10%) that includes noise

## Setting
- Metrics (note: we use the macro version of Precision, Recall and F1-Score to better consider workaround classes with fewer instances): 
    - Accuracy,  
    - Precision (macro), 
    - Recall (macro),
    - F1-score (macro)
- Encoding: binary encoded categorical attributes (note: we binary encode the categorical attributes because this encoding is less sparse than one-hot encoding)
- Validation method: split validation 
- Shuffling:
    - For general data set split (data set and pred set): yes
    - For data set split (train test): yes
    - For hyper-parameter optimisation: yes
    - For validation set per epoch (to perform early stopping): yes
- Hyper-parameter optimisation (HPO):
    - For the AE model: Tree-Structured Parzen Estimators approach (TPE) 
    - For the CNN model: Tree-Structured Parzen Estimators approach (TPE) 
    - Number of HPO runs: 10
    - Optimisation criteria: F1-score
- Data augmentation (i.e. replace noisy process instances by normal process instances): yes
- Threshold determination: depending on the event log the x'th percentile where x = {50, 60, 70, 80, 90, 100}

## Paper
This code repository belongs to a paper published in the Journal of Business Analytics.
If you use this code, please cite the paper as follows.

```
@article{weinzierl2021wdwdl,
    title={Detecting Temporal Workarounds in Business Processes - A Deep-Learning-based Method for Analysing Event Log Data},
    author={Sven Weinzierl and Verena Wolf and Tobias Pauli and Daniel Beverungen and Martin Matzner},
    journal={Journal of Business Analytics},
    year={2021}
}
```

You can access this paper [here](https://doi.org/10.1080/2573234X.2021.1978337).

A previous version of the JBA paper was published in the proceedings of the 28th European Conference on Information Systems (ECIS2020).

```
@proceedings{weinzierl2020wdwdl,
    title={Detecting Workarounds in Business Processes - a Deep Learning method for Analyzing Event Logs},
    author={Sven Weinzierl and Verena Wolf and Tobias Pauli and Daniel Beverungen and Martin Matzner},
    booktitle={Proceedings of the 28th European Conference on Information Systems (ECIS2020)},
    year={2020},
    pages={1-16}
    organization={AISeL}
}
```

You can access this paper [here](https://aisel.aisnet.org/ecis2020_rp/67/).





