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



