import numpy as np


def get_unique_events(process_instances_):
    flat_list = [item for sublist in process_instances_ for item in sublist]
    unique_events = np.unique(np.array(flat_list))

    return unique_events


def get_unique_context(process_instances_context_):
    num_context_attr = len(process_instances_context_[0][0])
    unique_context = []

    for index in range(0, num_context_attr):
        flat_list = [item[index] for sublist in process_instances_context_ for item in sublist]
        unique_context_attr = np.unique(np.array(flat_list))
        unique_context.append(unique_context_attr)

    return unique_context


def get_context_attributes_of_event(event):
    """ First context attribute is at the 4th position. """

    event = event.tolist()

    return event[3:]


def get_attribute_data_type(attribute_column):

    column_type = str(attribute_column.dtype)

    # column_type.startswith('int') or
    if column_type.startswith('float'):
        attribute_type = 'num'
    else:
        attribute_type = 'cat'

    return attribute_type


def get_encoding_mode(args, data_type):

    if data_type == 'num':
        mode = args.encoding_num

    elif data_type == 'cat':
        mode = args.encoding_cat

    return mode


def remove_end_mark_from_event_column(data):

    orig_column = data.drop(len(data) - 1)

    return orig_column
