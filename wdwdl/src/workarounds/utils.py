import random


def context_for_random_event(event, process_instances, process_instances_context):
    index = 0
    while True:
        process_instance_id = random.randrange(len(process_instances))
        if event in process_instances[process_instance_id]:
            return process_instances_context[process_instance_id][process_instances[process_instance_id].index(event)]
        if index == 25:
            return -1
        index = index + 1
