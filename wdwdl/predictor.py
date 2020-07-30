from __future__ import division
from keras.models import load_model
import wdwdl.utils as utils


def apply_wa_classification(args, data_set, preprocessor):

    number_attributes = preprocessor.data_structure['encoding']['num_values_features'] - 2  # case + time
    time_steps = preprocessor.data_structure['meta']['max_length_process_instance']
    data_set = data_set.reshape((data_set.shape[0], time_steps, number_attributes))

    model = load_model('%sclf_wa_mapping.h5' % args.checkpoint_dir, custom_objects={'f1_score': utils.f1_score})
    predictions = model.predict(data_set)

    return utils.arg_max(predictions)


def get_prediction_frequency(predictions):

    predictions = predictions.tolist()

    prediction_frequency = []
    unique_values = sorted(set(predictions))

    for value in unique_values:
        prediction_frequency.append({value: predictions.count(value)})

    return prediction_frequency