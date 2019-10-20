# Next event prediction with deep learning 

## Overview
- Input: 
  - event log with the attributes "process instance", "even", "timestamp", "context attribute 1", "...", "context attribute n"
  - all categorial attributes (despite the "timestamp attriubte") are integer-mapped and the timestamp attribute has the format "dd.mm.yyyy-hh:mm:ss".
- Pre-processing variants (so far, only variant 1 is implemented):
  1. A 3-dimensional tensor where on the third level the attribute "event" is one-hot encoded, further categorical context attributes are ordinal encoded (integer-mapped) und further numerical context attributes are min-max normalized (as is status) 
  2. Encoding / Embedding options
    1. Word Embedding (Word2Vec, Shallow Neural Network)
    2. Paragraph Embedding (Doc2Vec, Shallow Neural Network)
    --------------------
    3. Ordinal Encoding (Integer-Mapping) resp. OneHot Encoding (Baseline)
    4. Sum Encoding (proposed by Potdar and Pai (2017))
    5. Binary Encoding
    6. Hash Encoding (proposed by Mehdiyev and Fettke (2017)) 
    7. Helmet Encoding
    8. Backward Difference Encoding (proposed by Potdar and Pai (2017))
    9. Leave One Out Encoder (similar to target encoding/ M-esimate, but reduces the effect of outliers)
    10. Weight of Evidence
    
- Deep learning architectures
  1. Vanialla Long Short-Term Neural Network (Baseline) / Gated Recurrent Unit Neural Network (GRU)
  2. Stacked LSTM
  3. BI Directional LSTM
  4. CNN LSTM
  5. ConvLSTM
  6. Convolutional Neural Network (CNN) (maybe redundant to 8)
  7. Multi Layer Perceptron (MLP)
  8. Fully Convolutional Neural Network
  9. Residual Network (Resnet)
  10. Encoder
  11. Multi-scale Convolutional Neural Network (MCNN)
  12. Time Le-Net
  13. Multi Channel Deep Convolutional Neural Network (MCDCNN)
  14. Time-Convolutional Neural Network (Time-CNN)
  15. (Time Warping INvariant Echo State Network (TWIESW))
  16. (Differencial Neural Computer (DNC))
- Post processing: argmax


## Special remark on the differential neural computer
It is to mention that that our code of the dnc based not on the code (https://github.com/deepmind/dnc) from the original paper from Graves et al. 
("Hybrid computing using a neural network with dynamic external memory." Nature 538.7626 (2016): 471-476.), but
on an extension (https://github.com/thaihungle/MAED) implemented by Hung Le. The dnc consist of the files:
- controller.py
- dnc_v2.py
- feedforward_controller.py
- memory.py
- recurrent_controller.py
- utility.py

Note, there are two extensions to the original one. 
First, two controller (encoder and decoder) are used instead of only one controller. 
Second, the writing mechanism to the memory is protected. 


## Other useful repositories/sources
- https://github.com/verenich/ProcessSequencePrediction (CAiSE 2017)
- https://github.com/AdaptiveBProcess/GenerativeLSTM/tree/master/models (BPM 2019) 
- https://github.com/tnolle/binet (4 Papers)
- https://github.com/ProminentLab/DREAM-NAP
- http://contrib.scikit-learn.org/categorical-encoding/
- https://github.com/delas/plg (synthetic eventlog generator)
- https://github.com/keras-team/keras-contrib (keras extension)
- https://github.com/irhete/predictive-monitoring-thesis (hyperparameter optimization for lstms)
- https://docs.python-guide.org/writing/structure/ (structure of python project)
- https://keras.io/examples/babi_memnn/ (another DNC implementation that could be more efficient) 
