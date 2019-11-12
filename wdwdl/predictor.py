from __future__ import division
from keras.models import load_model
try:
    from itertools import izip as zip
except ImportError:
    pass
import wdwdl.utils as utils


def apply_wa_classification(args, data_set):
    model = load_model('%sclf_wa_mapping.h5' % args.checkpoint_dir)
    predictions = model.predict(data_set)
    return utils.arg_max(predictions)


