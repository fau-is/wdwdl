from __future__ import division
from keras.models import load_model
import wdwdl.utils as utils


def apply_wa_classification(args, data_set):

    model = load_model('%sclf_wa_mapping.h5' % args.checkpoint_dir)
    predictions = model.predict(data_set)

    return utils.arg_max(predictions)


def get_prediction_frequency(predictions):

    predictions = predictions.tolist()

    prediction_frequency = []
    unique_values = sorted(set(predictions))

    for value in unique_values:
        prediction_frequency.append({value: predictions.count(value)})

    return prediction_frequency