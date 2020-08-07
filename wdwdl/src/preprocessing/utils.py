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
