from __future__ import division
import wdwdl.src.utils.metric as metric
import wdwdl.src.utils.general as general
from tensorflow.keras.models import load_model


def apply_wa_classification(args, data_set, preprocessor, best_model_id):
    number_attributes = preprocessor.data_structure['encoding']['num_values_features'] - 2  # case + time
    time_steps = preprocessor.data_structure['meta']['max_length_process_instance']
    data_set = data_set.reshape((data_set.shape[0], time_steps, number_attributes))

    if args.hpo:
        model_name = '%sclf_wa_mapping_trial%s.h5' % (args.checkpoint_dir, best_model_id)
    else:
        model_name = '%sclf_wa_mapping.h5' % args.checkpoint_dir

    model = load_model(model_name, custom_objects={'f1_score': metric.f1_score})
    prob_dist = model.predict(data_set)
    pred = general.arg_max(prob_dist)

    return pred, prob_dist


def get_prediction_frequency(predictions):
    predictions = predictions.tolist()

    prediction_frequency = []
    unique_values = sorted(set(predictions))

    for value in unique_values:
        prediction_frequency.append({value: predictions.count(value)})

    return prediction_frequency
