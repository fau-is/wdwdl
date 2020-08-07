import numpy as np
import utils as utils


def substituted_activity(process_instance, context, unique_events, process_instances,
                         process_instances_context, max_substitutions=1):
    """
    Substitute an activity by another activity and its context attributes.
    :param process_instance:
    :param context:
    :param unique_events:
    :param process_instances:
    :param process_instances_context:
    :param max_substitutions:
    :return:
    """

    num_context_attr = len(context[0])
    unique_events = [x for x in unique_events if x not in process_instance]

    for index in range(0, max_substitutions):
        position = np.random.randint(0, len(process_instance) - 1)

        # event
        event_new = np.random.choice(unique_events)

        # context
        """
        select from a randomly selected process instance where the new event is included
        if the new event is not included in 25 randomly selected process instances, 
        than set the default value 0 for each context attribute
        """
        context_new = utils.get_context_for_random_event(event_new, process_instances, process_instances_context)
        if context_new == -1:
            context_new = [0] * num_context_attr

        if position > 0:
            process_instance = process_instance[:position] + [event_new] + process_instance[position:]
            context = context[:position] + [context_new] + context[position:]
        else:
            process_instance = [event_new] + process_instance[position:]
            context = [context_new] + context[position:]

    return process_instance, context


def injured_responsibility(process_instance, context, unique_context, max_injures=1):
    """
    Change the value of the resource attribute.
    The resource attribute is the first context attribute.
    :param process_instance:
    :param context:
    :param unique_context:
    :param max_injures:
    :return:
    """

    unique_resource = unique_context[0]

    for index in range(0, max_injures):
        position = np.random.randint(0, len(process_instance) - 1)
        unique_resource = [x for x in unique_resource if x != context[position][0]]
        context[position][0] = np.random.choice(unique_resource)

    return process_instance, context


def manipulated_data(process_instance, context, unique_context, max_events=1, max_attributes=1):
    """
    Change the value of data attributes.
    Data attributes are all context attributes without the resource attribute.
    :param process_instance:
    :param context:
    :param unique_context:
    :param max_events:
    :param max_attributes:
    :return:
    """

    num_attr = len(context[0])
    for index_event in range(0, max_events):

        for index_attr in range(0, max_attributes):
            # random event
            event = np.random.randint(0, len(process_instance) - 1)

            # random attribute
            attr = np.random.randint(1, num_attr)
            unique_context_attr = [x for x in unique_context[attr] if x != context[event][attr]]
            context[event][attr] = np.random.choice(unique_context_attr)

    return process_instance, context


def repeated_activity(process_instance, context, max_repetitions=1, max_repetition_length=5, min_repetition_length=3):
    """
    Repeat an activity and its context attributes n times.
    :param process_instance:
    :param context:
    :param max_repetitions:
    :param max_repetition_length:
    :param min_repetition_length:
    :return:
    """

    for index in range(0, max_repetitions):

        start = np.random.randint(0, len(process_instance) - 1)
        end = start + 1
        number_repetitions = np.random.randint(min_repetition_length, max_repetition_length + 1)

        if start > 0:
            process_instance = process_instance[:start] + [
                process_instance[start]] * number_repetitions + process_instance[end - 1:]
            context = context[:start] + [context[start]] * number_repetitions + context[end - 1:]
        else:
            process_instance = [process_instance[start]] * number_repetitions + process_instance[end - 1:]
            context = [context[start]] * number_repetitions + context[end - 1:]

    return process_instance, context


def interchanged_activity(process_instance, context, max_interchanges=1):
    """
    Pairwise change of two activities and its context attributes.
    :param process_instance:
    :param context:
    :param max_interchanges:
    :return:
    """

    for index in range(0, max_interchanges):

        start = np.random.randint(0, len(process_instance) - 1)
        end = start + 1

        if start > 0:
            process_instance = process_instance[:start] + [process_instance[end]] + [
                process_instance[start]] + process_instance[end:]
            context = context[:start] + [context[end]] + [context[start]] + context[end:]
        else:
            process_instance = [process_instance[end]] + [process_instance[start]] + process_instance[end:]
            context = [context[end]] + [context[start]] + context[end:]

    return process_instance, context


def bypassed_activity(process_instance, context, max_sequence_size=1):
    """
    Skips an activity or an sequence of of activities and its context attributes.
    :param process_instance:
    :param context:
    :param max_sequence_size:
    :return:
    """

    size = np.random.randint(1, min(len(process_instance) - 1, max_sequence_size) + 1)
    start = np.random.randint(0, len(process_instance) - size)
    end = start + size

    process_instance = process_instance[:start] + process_instance[end:]
    context = context[:start] + context[end:]

    return process_instance, context


def added_activity(process_instance, context, unique_events, process_instances, process_instances_context, max_adds=1):
    """
    Adds an activity and its context attributes.
    :param process_instance:
    :param context:
    :param unique_events:
    :param process_instances:
    :param process_instances_context:
    :param max_adds:
    :return:
    """

    num_context_attr = len(context[0])
    unique_events = [x for x in unique_events if x not in process_instance]

    for index in range(0, max_adds):
        position = np.random.randint(0, len(process_instance) - 1)

        # event
        event_new = np.random.choice(unique_events)

        # context
        """
        select from a randomly selected process instance where the new event is included
        if the new event is not included in 25 randomly selected process instances, 
        than set the default value 0 for each context attribute
        """
        context_new = utils.get_context_for_random_event(event_new, process_instances, process_instances_context)
        if context_new == -1:
            context_new = [0] * num_context_attr

        if position > 0:
            process_instance = process_instance[:position] + [event_new] + process_instance[position - 1:]
            context = context[:position] + [context_new] + context[position - 1:]
        else:
            process_instance = [event_new] + process_instance
            context = [context_new] + context

    return process_instance, context
