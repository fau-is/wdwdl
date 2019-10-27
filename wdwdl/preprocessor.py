from __future__ import division
import csv
import numpy
import pandas
import category_encoders
import copy
import wdwdl.utils as utils
import sklearn
from sklearn.model_selection import KFold, ShuffleSplit
import tensorflow as tf
import keras
import plotly
from matplotlib import pyplot as plt


class Preprocessor(object):
    data_structure = {
        'support': {
            'num_folds': 1,
            'data_dir': "",
            'encoded_data_dir': "",
            'ascii_offset': 161,
            'data_format': "%d.%m.%Y-%H:%M:%S",
            'train_index_per_fold': [],
            'test_index_per_fold': [],
            'iteration_cross_validation': 0,
            'elements_per_fold': 0,
            'event_labels': [],
            'event_types': [],
            'map_event_label_to_event_id': [],
            'map_event_id_to_event_label': [],
            'map_event_type_to_event_id': [],
            'map_event_id_to_event_type': [],
            'end_process_instance': '!'
        },

        'encoding': {
            'eventlog_df': pandas.DataFrame,
            'event_ids': {},
            'context_attributes': [],  # lengths of encoded attributes
            'num_values_control_flow': 0,
            'num_values_context': 0,
            'num_values_features': 0
        },

        'meta': {
            'num_features': 0,
            'num_event_ids': 0,
            'max_length_process_instance': 0,
            'num_attributes_context': 0,
            'num_attributes_control_flow': 3,  # process instance id, event id and timestamp
            'num_process_instances': 0
        },

        'data': {
            'process_instances': [],
            'ids_process_instances': [],
            'context_attributes_process_instances': [],
            'process_instances_raw': [],
            'context_attributes_process_instances_raw': [],
            'train': {
                'process_instances': [],
                'context_attributes': [],
                'event_ids': [],
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            },

            'test': {
                'process_instances': [],
                'context_attributes': [],
                'event_ids': []
            }
        }
    }

    def __init__(self, args):
        utils.llprint("Initialization ... \n")
        self.data_structure['support']['num_folds'] = args.num_folds
        self.data_structure['support']['data_dir'] = args.data_dir + args.data_set
        self.data_structure['support']['encoded_data_dir'] = r'%s' % args.data_dir + r'encoded_%s' % args.data_set

        eventlog_df = pandas.read_csv(self.data_structure['support']['data_dir'], sep=';')
        self.data_structure['encoding']['eventlog_df'] = eventlog_df
        eventlog_df = self.encode_eventlog(args, eventlog_df)

        self.set_number_control_flow_attributes()

        self.get_sequences_from_encoded_eventlog(eventlog_df)

        self.data_structure['support']['elements_per_fold'] = \
            int(round(self.data_structure['meta']['num_process_instances'] /
                      self.data_structure['support']['num_folds']))

        end_marked_process_instances = []
        for process_instance in self.data_structure['data']['process_instances']:
            process_instance.append(self.get_encoded_end_mark())
            end_marked_process_instances.append(process_instance)

        self.data_structure['data']['process_instances'] = end_marked_process_instances

        self.set_max_length_process_instance()

        self.data_structure['support']['event_labels'] = list(
            self.data_structure['encoding']['event_ids']['mapping'].values())

        self.data_structure['support']['event_types'] = copy.copy(self.data_structure['support']['event_labels'])
        self.data_structure['support']['map_event_label_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_id_to_event_label'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_type_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_types']))
        self.data_structure['support']['map_event_id_to_event_type'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_types']))

        self.data_structure['meta']['num_event_ids'] = len(self.data_structure['support']['event_labels'])
        self.data_structure['meta']['num_features'] = self.data_structure['meta']['num_event_ids'] + \
                                                      self.data_structure['meta']['num_attributes_context']
        self.set_number_values_features()

    def set_number_values_features(self):
        self.data_structure['encoding']['num_values_features'] = self.data_structure['encoding'][
                                                                     'num_values_control_flow'] + \
                                                                 self.data_structure['encoding']['num_values_context']
    def set_number_control_flow_attributes(self):
        # + 2 -> case + time
        self.data_structure['encoding']['num_values_control_flow'] = \
            2 + self.data_structure['encoding']['event_ids']['length']

    def set_max_length_process_instance(self):
        self.data_structure['meta']['max_length_process_instance'] = max(
            [len(x) for x in self.data_structure['data']['process_instances']])

    def split_event_log(self, args):

        if args.cross_validation:
            self.set_indices_k_fold_validation()
        else:
            self.set_indices_split_validation(args)

    def encode_eventlog(self, args, eventlog_df):

        # case
        encoded_eventlog_df = pandas.DataFrame(eventlog_df.iloc[:, 0])

        for column_name in eventlog_df:
            column_index = eventlog_df.columns.get_loc(column_name)

            # skip case and timestamp
            if column_index == 1 or column_index > 2:

                column = eventlog_df[column_name]
                column_data_type = self.get_attribute_data_type(column)  # cat or num

                if column_index == 1:
                    # event ID
                    if column_data_type == 'num':
                        # TODO: can integer-mapped events possibly be encoded with onehot, hash, bin?
                        column_with_end_mark = self.add_end_mark_to_event_column(column_name)[column_name]
                        self.save_mapping_of_encoded_events(column_with_end_mark, column_with_end_mark)
                        encoded_column = column

                    elif column_data_type == 'cat':
                        encoded_column = self.encode_column(args, 'event', column_name, column_data_type)

                    if isinstance(encoded_column, pandas.DataFrame):
                        self.set_length_of_event_encoding(len(encoded_column.columns))
                    elif isinstance(encoded_column, pandas.Series):
                        self.set_length_of_event_encoding(1)

                else:
                    # context attribute
                    encoded_column = self.encode_column(args, 'context', column_name, column_data_type)

                encoded_eventlog_df = encoded_eventlog_df.join(encoded_column)

            else:
                encoded_eventlog_df[column_name] = eventlog_df[column_name]

        encoded_eventlog_df.to_csv(self.data_structure['support']['encoded_data_dir'], sep=';', index=False)

        return encoded_eventlog_df

    def get_attribute_data_type(self, attribute_column):

        column_type = str(attribute_column.dtype)

        # column_type.startswith('int') or
        if column_type.startswith('float'):
            attribute_type = 'num'
        else:
            attribute_type = 'cat'

        return attribute_type

    def get_encoding_mode(self, args, data_type):

        if data_type == 'num':
            mode = args.encoding_num

        elif data_type == 'cat':
            mode = args.encoding_cat

        return mode

    def encode_column(self, args, attribute_type, attribute_name, column_data_type):

        mode = self.get_encoding_mode(args, column_data_type)

        if mode == 'int':
            encoded_column = self.apply_integer_mapping(attribute_type, attribute_name)

        elif mode == 'min_max_norm':
            encoded_column = self.apply_min_max_normalization(attribute_name)

        elif mode == 'onehot':
            encoded_column = self.apply_one_hot_encoding(attribute_type, attribute_name)

        elif mode == 'bin':
            encoded_column = self.apply_binary_encoding(attribute_type, attribute_name)

        elif mode == 'hash':
            encoded_column = self.apply_hash_encoding(args, attribute_type, attribute_name)

        else:
            # no encoding
            encoded_column = self.data_structure['encoding']['eventlog_df'][attribute_name]

        return encoded_column

    def apply_integer_mapping(self, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        if attribute_type == 'event':
            dataframe = self.add_end_mark_to_event_column(column_name)

        data = dataframe[column_name].fillna("missing")
        unique_values = data.unique().tolist()
        int_mapping = dict(zip(unique_values, range(len(unique_values))))
        encoded_data = data.map(int_mapping)

        if attribute_type == 'event':
            self.save_mapping_of_encoded_events(data, encoded_data)
            encoded_data = self.remove_end_mark_from_event_column(encoded_data)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        return encoded_data

    def apply_min_max_normalization(self, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        data = dataframe[column_name].fillna(dataframe[column_name].mean())
        encoded_data = data / data.max()

        self.set_length_of_context_encoding(1)

        return encoded_data

    def apply_one_hot_encoding(self, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        if attribute_type == 'event':
            dataframe = self.add_end_mark_to_event_column(column_name)

        data = dataframe[column_name]
        unique_values = data.unique().tolist()
        encoded_data_row = []
        encoded_data_rows = []

        for data_row in data:

            for value in unique_values:

                if value == data_row:
                    encoded_data_row.append(1.0)
                else:
                    encoded_data_row.append(0.0)

            encoded_data_rows.append(encoded_data_row)
            encoded_data_row = []

        encoded_data = pandas.DataFrame(encoded_data_rows)

        new_column_names = []
        for value in unique_values:
            new_column_names.append(column_name + "_%s" % value)

        encoded_data = encoded_data.rename(columns=dict(zip(encoded_data.columns.tolist(), new_column_names)))

        if attribute_type == 'event':
            self.save_mapping_of_encoded_events(data, encoded_data)
            encoded_data = self.remove_end_mark_from_event_column(encoded_data)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        return encoded_data

    def apply_binary_encoding(self, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        if attribute_type == 'event':
            dataframe = self.add_end_mark_to_event_column(column_name)

        binary_encoder = category_encoders.BinaryEncoder(cols=[column_name])
        encoded_df = binary_encoder.fit_transform(dataframe)

        encoded_data = encoded_df[
            encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        if attribute_type == 'event':
            self.save_mapping_of_encoded_events(dataframe[column_name], encoded_data)
            encoded_data = self.remove_end_mark_from_event_column(encoded_data)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        return encoded_data

    def apply_hash_encoding(self, args, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        if attribute_type == 'event':
            dataframe = self.add_end_mark_to_event_column(column_name)

        dataframe[column_name] = dataframe[column_name].fillna("missing")
        hash_encoder = category_encoders.HashingEncoder(cols=[column_name], n_components=args.num_hash_output)
        encoded_df = hash_encoder.fit_transform(dataframe)

        encoded_data = encoded_df[encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith('col_')]]

        new_column_names = []
        for number in range(len(encoded_df.columns)):
            new_column_names.append(column_name + "_%d" % number)

        encoded_data = encoded_data.rename(columns=dict(zip(encoded_df.columns.tolist(), new_column_names)))

        if attribute_type == 'event':
            self.save_mapping_of_encoded_events(dataframe[column_name], encoded_data)
            encoded_data = self.remove_end_mark_from_event_column(encoded_data)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        return encoded_data

    def add_end_mark_to_event_column(self, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']
        end_mark = self.data_structure['support']['end_process_instance']

        df_columns = dataframe.columns
        new_row = []
        for column in df_columns:
            if column == column_name:
                new_row.append(end_mark)
            else:
                new_row.append(0)

        row_df = pandas.DataFrame([new_row], columns=dataframe.columns)
        dataframe = dataframe.append(row_df, ignore_index=True)

        return dataframe

    def remove_end_mark_from_event_column(self, data):

        orig_column = data.drop(len(data) - 1)

        return orig_column

    def save_mapping_of_encoded_events(self, column, encoded_column):

        encoded_column_tuples = []
        for entry in encoded_column.values.tolist():

            if type(entry) != list:
                encoded_column_tuples.append((entry,))
            else:
                encoded_column_tuples.append(tuple(entry))

        tuple_all_rows = list(zip(column.values.tolist(), encoded_column_tuples))

        tuple_unique_rows = []
        for tuple_row in tuple_all_rows:

            if tuple_row not in tuple_unique_rows:
                tuple_unique_rows.append(tuple_row)

        mapping = dict(tuple_unique_rows)

        self.data_structure['encoding']['event_ids']['mapping'] = mapping

    def get_encoded_end_mark(self):

        return self.data_structure['encoding']['event_ids']['mapping'][
            self.data_structure['support']['end_process_instance']]

    def set_length_of_event_encoding(self, num_columns):
        self.data_structure['encoding']['event_ids']['length'] = num_columns

    def set_length_of_context_encoding(self, num_columns):
        self.data_structure['encoding']['context_attributes'].append(num_columns)

    def get_sequences_from_raw_eventlog(self, eventlog_df):
        id_latest_process_instance_raw = ''
        process_instance_raw = ''
        first_event_of_process_instance_raw = True
        context_attributes_process_instance_raw = []

        for index, event in eventlog_df.iterrows():

            id_current_process_instance = event[0]

            if id_current_process_instance != id_latest_process_instance_raw:
                id_latest_process_instance_raw = id_current_process_instance

                if not first_event_of_process_instance_raw:
                    self.add_data_to_data_structure(process_instance_raw, 'process_instances_raw')

                    if self.data_structure['meta']['num_attributes_context'] > 0:
                        self.add_data_to_data_structure(context_attributes_process_instance_raw,
                                                        'context_attributes_process_instances_raw')

                process_instance_raw = []

                if self.data_structure['meta']['num_attributes_context'] > 0:
                    context_attributes_process_instance_raw = []

            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_event = self.get_context_attributes_of_event(event)
                context_attributes_process_instance_raw.append(context_attributes_event)

            process_instance_raw.append(event[1])
            first_event_of_process_instance_raw = False

        self.add_data_to_data_structure(process_instance_raw, 'process_instances_raw')

        if self.data_structure['meta']['num_attributes_context'] > 0:
            self.add_data_to_data_structure(context_attributes_process_instance_raw,
                                            'context_attributes_process_instances_raw')

    def get_sequences_from_encoded_eventlog(self, eventlog_df):

        id_latest_process_instance = ''
        process_instance = ''
        first_event_of_process_instance = True
        context_attributes_process_instance = []
        output = True

        for index, event in eventlog_df.iterrows():

            id_current_process_instance = event[0]
            if output:
                self.check_for_context_attributes_df(event)
                output = False

            if id_current_process_instance != id_latest_process_instance:
                self.add_data_to_data_structure(id_current_process_instance, 'ids_process_instances')
                id_latest_process_instance = id_current_process_instance

                if not first_event_of_process_instance:
                    self.add_data_to_data_structure(process_instance, 'process_instances')

                    if self.data_structure['meta']['num_attributes_context'] > 0:
                        self.add_data_to_data_structure(context_attributes_process_instance,
                                                        'context_attributes_process_instances')

                process_instance = []

                if self.data_structure['meta']['num_attributes_context'] > 0:
                    context_attributes_process_instance = []

                self.data_structure['meta']['num_process_instances'] += 1

            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_event = self.get_encoded_context_attributes_of_event(event)
                context_attributes_process_instance.append(context_attributes_event)

            process_instance = self.add_encoded_event_to_process_instance(event, process_instance)
            first_event_of_process_instance = False

        self.add_data_to_data_structure(process_instance, 'process_instances')

        if self.data_structure['meta']['num_attributes_context'] > 0:
            self.add_data_to_data_structure(context_attributes_process_instance, 'context_attributes_process_instances')
        self.data_structure['meta']['num_process_instances'] += 1

    def check_for_context_attributes_df(self, event):

        if len(event) == self.data_structure['encoding']['num_values_control_flow']:
            utils.llprint("No context attributes found ...\n")
        else:
            self.data_structure['meta']['num_attributes_context'] = len(
                self.data_structure['encoding']['context_attributes'])
            self.data_structure['encoding']['num_values_context'] = sum(
                self.data_structure['encoding']['context_attributes'])
            utils.llprint("%d context attributes found ...\n" % self.data_structure['meta']['num_attributes_context'])

    def add_encoded_event_to_process_instance(self, event, process_instance):

        encoded_event_id = []
        start_index = 1
        end_index = self.data_structure['encoding']['event_ids']['length'] + 1

        for enc_val in range(start_index, end_index):
            encoded_event_id.append(event[enc_val])

        process_instance.append(tuple(encoded_event_id))

        return process_instance

    def get_context_attributes_of_event(self, event):
        """ First context attribute is at the 4th position. """

        event = event.tolist()

        return event[3:]

    def get_encoded_context_attributes_of_event(self, event):

        event = event.tolist()
        context_attributes_event = []

        for context_attribute_index in range(self.data_structure['encoding']['num_values_control_flow'],
                                             self.data_structure['encoding']['num_values_control_flow'] +
                                             self.data_structure['encoding']['num_values_context']):
            context_attributes_event.append(event[context_attribute_index])

        return context_attributes_event

    def add_data_to_data_structure(self, values, structure):

        self.data_structure['data'][structure].append(values)

    def set_training_set(self):

        utils.llprint("Get training instances ... \n")
        process_instances_train, context_attributes_train, _ = self.set_instances_of_fold('train')

        utils.llprint("Create cropped training instances ... \n")
        cropped_process_instances, cropped_context_attributes, next_events = \
            self.get_cropped_instances(
                process_instances_train,
                context_attributes_train)

        utils.llprint("Create training set data as tensor ... \n")
        features_data = self.get_data_tensor(cropped_process_instances,
                                             cropped_context_attributes,
                                             'train')

        utils.llprint("Create training set label as tensor ... \n")
        labels = self.get_label_tensor(cropped_process_instances,
                                       next_events)

        self.data_structure['data']['train']['features_data'] = features_data
        self.data_structure['data']['train']['labels'] = labels

    def get_event_type(self, predictions):

        max_prediction = 0
        event_type = ''
        index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                event_type = self.data_structure['support']['map_event_id_to_event_type'][index]
            index += 1

        return event_type


    def split_validation(self, data_set, label):
        return sklearn.model_selection.train_test_split(data_set, label, test_size=0.3, random_state=0)


    def set_indices_k_fold_validation(self):
        """ Produces indices for each fold of a k-fold cross-validation. """

        k_fold = KFold(n_splits=self.data_structure['support']['num_folds'], random_state=0, shuffle=False)

        for train_indices, test_indices in k_fold.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    def set_indices_split_validation(self, args):
        """ Produces indices for train and test set of a split-validation. """

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test, random_state=0)

        for train_indices, test_indices in shuffle_split.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    def set_instances_of_fold(self, mode):
        """ Retrieves instances of a fold. """

        process_instances_of_fold = []
        context_attributes_of_fold = []
        event_ids_of_fold = []

        for value in self.data_structure['support'][mode + '_index_per_fold'][
            self.data_structure['support']['iteration_cross_validation']]:
            process_instances_of_fold.append(self.data_structure['data']['process_instances'][value])
            event_ids_of_fold.append(self.data_structure['data']['ids_process_instances'][value])

            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_of_fold.append(
                    self.data_structure['data']['context_attributes_process_instances'][value])

        if mode == 'train':
            self.data_structure['data']['train']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['train']['context_attributes'] = context_attributes_of_fold
            self.data_structure['data']['train']['event_ids'] = event_ids_of_fold

        elif mode == 'test':
            self.data_structure['data']['test']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['test']['context_attributes'] = context_attributes_of_fold
            self.data_structure['data']['test']['event_ids'] = event_ids_of_fold

        return process_instances_of_fold, context_attributes_of_fold, event_ids_of_fold

    def get_cropped_instances(self, process_instances, context_attributes_process_instances):
        """ Crops prefixes out of instances. """

        cropped_process_instances = []
        cropped_context_attributes = []
        next_events = []

        if self.data_structure['meta']['num_attributes_context'] > 0:

            for process_instance, context_attributes_process_instance in zip(process_instances,
                                                                             context_attributes_process_instances):
                for i in range(0, len(process_instance)):

                    if i == 0:
                        continue

                    # 0:i -> get 0 up to n-1 events of a process instance, since n is the label
                    cropped_process_instances.append(process_instance[0:i])
                    cropped_context_attributes.append(context_attributes_process_instance[0:i])
                    # label
                    next_events.append(process_instance[i])
        else:
            for process_instance in process_instances:
                for i in range(0, len(process_instance)):

                    if i == 0:
                        continue
                    cropped_process_instances.append(process_instance[0:i])
                    # label
                    next_events.append(process_instance[i])

        return cropped_process_instances, cropped_context_attributes, next_events

    def get_cropped_instance(self, prefix_size, index, process_instance):
        """ Crops prefixes out of a single process instance. """

        cropped_process_instance = process_instance[:prefix_size]
        if self.data_structure['meta']['num_attributes_context'] > 0:
            cropped_context_attributes = self.data_structure['data']['test']['context_attributes'][index][:prefix_size]
        else:
            cropped_context_attributes = []

        return cropped_process_instance, cropped_context_attributes

    def get_data_tensor(self, cropped_process_instances, cropped_context_attributes_process_instance, mode):
        """ Produces a vector-oriented representation of data as 3-dimensional tensor. """

        if mode == 'train':
            data_set = numpy.zeros((
                len(cropped_process_instances),
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['encoding']['event_ids']['length'] + self.data_structure['encoding'][
                    'num_values_context']), dtype=numpy.float64)
        else:
            data_set = numpy.zeros((
                1,
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['encoding']['event_ids']['length'] + self.data_structure['encoding'][
                    'num_values_context']), dtype=numpy.float32)

        # ToDo: do not create a single numpy array for all rows of data,
        #  maybe we can find a workaround for this
        for index, cropped_process_instance in enumerate(cropped_process_instances):

            left_pad = self.data_structure['meta']['max_length_process_instance'] - len(cropped_process_instance)

            if self.data_structure['meta']['num_attributes_context'] > 0:
                cropped_context_attributes = cropped_context_attributes_process_instance[index]

            for time_step, event in enumerate(cropped_process_instance):
                for tuple_idx in range(0, self.data_structure['encoding']['event_ids']['length']):
                    data_set[index, time_step + left_pad, tuple_idx] = event[tuple_idx]

                if self.data_structure['meta']['num_attributes_context'] > 0:
                    for context_attribute_index in range(0, self.data_structure['encoding']['num_values_context']):
                        data_set[index, time_step + left_pad, self.data_structure['encoding']['event_ids'][
                            'length'] + context_attribute_index] = cropped_context_attributes[time_step][
                            context_attribute_index]

        return data_set

    def get_data_tensor_for_single_prediction(self, args, cropped_process_instance, cropped_context_attributes):

        data_set = self.get_data_tensor(
            [cropped_process_instance],
            [cropped_context_attributes],
            'test')

        # Change structure if CNN LSTM
        if args.dnn_architecture == 3:
            data_set = data_set.reshape((
                data_set.shape[0], 1,
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_values_features']))

        # Change structure if ConvLSTM
        if args.dnn_architecture == 4:
            data_set = data_set.reshape((
                1, 1, 1,
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_values_features']))

        return data_set

    def get_label_tensor(self, cropped_process_instances, next_events):
        """ Produces a vector-oriented representation of label as 2-dimensional tensor. """

        label = numpy.zeros((len(cropped_process_instances), len(self.data_structure['support']['event_types'])),
                            dtype=numpy.float64)

        for index, cropped_process_instance in enumerate(cropped_process_instances):

            for event_type in self.data_structure['support']['event_types']:

                if event_type == next_events[index]:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 1
                else:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 0

        return label


    def get_2d_data_tensor(self):

        process_instances = self.data_structure['data']['process_instances']
        context_attributes_process_instances = self.data_structure['data']['context_attributes_process_instances']
        number_context_attributes = self.data_structure['encoding']['num_values_context']
        number_control_flow_attributes = self.data_structure['encoding']['num_values_control_flow'] - 2  # case + time
        number_attributes = self.data_structure['encoding']['num_values_features'] - 2  # case + time
        vector_length = self.data_structure['meta']['max_length_process_instance'] * number_attributes

        # create structure
        data_set = numpy.zeros((
            len(process_instances),
            vector_length
        ))

        # fill data
        for index_instance in range(0, len(process_instances)):
            for time_step in range(0, len(process_instances[index_instance]) - 1):  # -1 end marking

                # event
                event_attributes = list(process_instances[index_instance][time_step])
                number_event_attributes = len(event_attributes)

                for index_attribute in range(0, number_event_attributes):
                    data_set[index_instance, time_step * (
                                number_context_attributes + number_event_attributes) + index_attribute] = \
                        event_attributes[index_attribute]

                # context

                context_attributes = context_attributes_process_instances[index_instance][time_step]
                number_context_attributes = len(context_attributes)

                for index_attribute in range(0, number_context_attributes):
                    data_set[index_instance, time_step * (
                                number_context_attributes + 1) + number_control_flow_attributes + index_attribute] = \
                        context_attributes[index_attribute]

        return data_set

    def plot_learning_curve(self, history, learning_epochs):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.figure()
        plt.plot(range(learning_epochs), loss, 'bo', label='Training loss')
        plt.plot(range(learning_epochs), val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    def plot_reconstruction_error(self, df, col):

        x = df.index.tolist()
        y = df[col].tolist()

        trace = {'type': 'scatter',
                 'x': x,
                 'y': y,
                 'mode': 'markers'
                 # 'marker': {'colorscale': 'red', 'opacity': 0.5}
                 }
        data = plotly.graph_objs.Data([trace])
        layout = {'title': 'Reconstruction error for each process instance',
                  'titlefont': {'size': 30},
                  'xaxis': {'title': 'Process instance', 'titlefont': {'size': 20}},
                  'yaxis': {'title': 'Reconstruction error', 'titlefont': {'size': 20}},
                  'hovermode': 'closest'
                  }
        figure = plotly.graph_objs.Figure(data=data, layout=layout)
        return figure

    def clean_event_log(self, args):
        """ clean the event log with an autoencoder. """

        utils.llprint("Create data set as tensor ... \n")
        features_data = self.get_2d_data_tensor()
        features_data_df = pandas.DataFrame(data=features_data[0:, 0:],
                                            index=[i for i in range(features_data.shape[0])],
                                            columns=['f' + str(i) for i in range(features_data.shape[1])])

        # autoencoder
        input_dimension = features_data.shape[1]
        encoding_dimension = 100
        learning_epochs = 100

        input_layer = keras.layers.Input(shape=(input_dimension,))
        encoder = keras.layers.Dense(encoding_dimension, activation='tanh')(input_layer)
        encoder = keras.layers.Dense(int(encoding_dimension), activation='tanh')(encoder)
        encoder = keras.layers.Dense(int(encoding_dimension / 2), activation='tanh')(encoder)
        encoder = keras.layers.Dense(int(encoding_dimension / 4), activation='tanh')(encoder)
        decoder = keras.layers.Dense(int(encoding_dimension / 2), activation='tanh')(encoder)
        decoder = keras.layers.Dense(int(encoding_dimension), activation='tanh')(decoder)
        decoder = keras.layers.Dense(int(encoding_dimension), activation='tanh')(decoder)
        decoder = keras.layers.Dense(input_dimension, activation='tanh')(decoder)

        autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        autoencoder.compile(optimizer='adam', loss='mse')

        history = autoencoder.fit(features_data, features_data,
                                  epochs=learning_epochs,
                                  batch_size=256,
                                  shuffle=True,
                                  validation_split=0.1,
                                  )

        # df_history = pandas.DataFrame(history.history)

        # self.plot_learning_curve(history, learning_epochs)

        # remove noise of event log data
        predictions = autoencoder.predict(features_data)
        mse = numpy.mean(numpy.power(features_data - predictions, 2), axis=1)
        df_error = pandas.DataFrame({'reconstruction_error': mse}, index=[i for i in range(features_data.shape[0])])

        plot_error = self.plot_reconstruction_error(df_error, 'reconstruction_error')
        # print(plotly.offline.plot(plot_error))
        no_outliers = df_error.index[df_error['reconstruction_error'] <= 0.2].tolist()
        # features_data = features_data_df.drop(features_data_df.index[outliers])

        return no_outliers

    def workaround_injured_responsiblity(self, process_instance, context, unique_context, max_injures=1):
        """
        Change the value of the resource attribute.
        The resource attribute is the first context attribute.
        """

        unique_resource = unique_context[0]


        for index in range(0, max_injures):
            position = numpy.random.randint(0, len(process_instance) - 1)
            unique_resource = [x for x in unique_resource if x != context[position][0]]
            context[position][0] = numpy.random.choice(unique_resource)

        return process_instance, context


    def workaround_manipulated_data(self, process_instance, context, unique_context, max_events=1, max_attributes=1):
        """
        Change the value of data attributes.
        Data attributes are all context attributes without the resource attribute.
        """

        num_attr = len(context[0])
        for index_event in range(0, max_events):

            for index_attr in range(0, max_attributes):
                # random event
                event = numpy.random.randint(0, len(process_instance) - 1)

                # random attribute
                attr = numpy.random.randint(1, num_attr)
                unique_context_attr = [x for x in unique_context[attr] if x != context[event][attr]]
                context[event][attr] = numpy.random.choice(unique_context_attr)

        return process_instance, context


    def workaround_repeated_activity(self, process_instance, context, max_repetitions=1, max_repetition_length=5):
        """
        Repeat an activity and its context attributes n times.
        """

        for index in range(0, max_repetitions):

            start = numpy.random.randint(0, len(process_instance)-1)
            end = start + 1

            if start > 0:
                process_instance = process_instance[:start] + [process_instance[start]] * max_repetition_length + process_instance[end-1:]
                context = context[:start] + [context[start]] * max_repetition_length + context[end-1:]
            else:
                process_instance = [process_instance[start]] * max_repetition_length + process_instance[end-1:]
                context = [context[start]] * max_repetition_length + context[end-1:]

        return process_instance, context


        return process_instance, context

    def workaround_substitued_activity(self, process_instance, context, max_substitutions=1):
        """
        Substitute an activity by another activity and its context attributes.
        """

        return process_instance, context

    def workaround_interchanged_activity(self, process_instance, context, max_interchanges=1):
        """
        Pairwise change of two activities and its context attributes.
        """

        for index in range(0, max_interchanges):

            start = numpy.random.randint(0, len(process_instance)-1)
            end = start + 1

            if start > 0:
                process_instance = process_instance[:start] + [process_instance[end]] + [process_instance[start]] + process_instance[end:]
                context = context[:start] + [context[end]] + [context[start]] + context[end:]
            else:
                process_instance = [process_instance[end]] + [process_instance[start]] + process_instance[end:]
                context = [context[end]] + [context[start]] + context[end:]

        return process_instance, context

    def workaround_bypassed_activity(self, process_instance, context, max_sequence_size=1):
        """
        Skips an activity or an sequence of of activities and its context attributes.
        """

        size = numpy.random.randint(1, min(len(process_instance) - 1, max_sequence_size) + 1)
        start = numpy.random.randint(0, len(process_instance) - size)
        end = start + size

        process_instance = process_instance[:start] + process_instance[end:]
        context = context[:start] + context[end:]

        return process_instance, context

    def workaround_added_activity(self, process_instance, context, max_adds=1):

        return process_instance, context

    def add_workarounds_to_event_log(self, args, no_outliers):

        # get process instances
        eventlog_df = self.data_structure['encoding']['eventlog_df']
        self.get_sequences_from_raw_eventlog(eventlog_df)
        process_instances = self.data_structure['data']['process_instances_raw']
        process_instances_context = self.data_structure['data']['context_attributes_process_instances_raw']

        # get process instance without noise
        process_instances_ = []
        process_instances_context_ = []
        for index in range(0, len(process_instances)):
            if index in no_outliers:
                process_instances_.append(process_instances[index])
                process_instances_context_.append(process_instances_context[index])

        # add workaround
        process_instances_wa = []
        process_instances_context_wa = []
        unique_events = []
        unique_context = []
        label = [0] * len(process_instances_)  # 0 means that a process instance does not include a workaround
        probability = 0.9  # 30% of the process instances include workarounds
        unique_events = utils.get_unique_events(process_instances_)
        unique_context = utils.get_unique_context(process_instances_context_)

        for index in range(0, len(process_instances_)):

            if numpy.random.uniform(0, 1) <= probability and len(process_instances_[index]) >= 2:

                workaround_form = int(numpy.random.uniform(1, 7))

                if workaround_form == 1:  # "injured_responsibility"
                    process_instance_wa, process_instance_context_wa = \
                        self.workaround_injured_responsiblity(
                            process_instances_[index],
                            process_instances_context_[index],
                            unique_context,
                            max_injures=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 2:  # "manipulated_data"
                    process_instance_wa, process_instance_context_wa = \
                        self.workaround_manipulated_data(
                            process_instances_[index],
                            process_instances_context_[index],
                            unique_context,
                            max_events=1,
                            max_attributes=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 3:  # "repeated_activity"
                    process_instance_wa, process_instance_context_wa = \
                        self.workaround_repeated_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_repetitions=1,
                            max_repetition_length=10
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 4:  # "substituted_activity"
                    process_instance_wa, process_instance_context_wa = \
                        self.workaround_substitued_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_substitutions=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 5:  # "interchanged_activity"
                    process_instance_wa, process_instance_context_wa = \
                        self.workaround_interchanged_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_interchanges=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 6:  # "bypassed_activity"
                    process_instance_wa, process_instance_context_wa = \
                        self.workaround_bypassed_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_sequence_size=3
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 7:  # "added_activity"
                    process_instance_wa, process_instance_context_wa = \
                        self.workaround_added_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_adds=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

            else:
                process_instances_wa.append(process_instances_[index])
                process_instances_context_wa.append(process_instances_context_[index])

        # from instance-based list to event-based numpy array
        number_of_events = sum(list([len(element) for element in process_instances_wa]))
        data_set = numpy.zeros((number_of_events, 3 + len(process_instances_context_wa[0][0])))  # case, event and time

        index_ = 0
        for index in range(0, len(process_instances_wa)):

            for index_event in range(0, len(process_instances_wa[index])):
                # case
                data_set[index_, 0] = index

                # event
                data_set[index_, 1] = process_instances_wa[index][index_event]

                # time
                data_set[index_, 2] = 0  # only for filling the gap

                # context attributes
                for index_context in range(0, len(process_instances_context_wa[index][index_event])):
                    data_set[index_, index_context + 3] = process_instances_context_wa[index][index_event][
                        index_context]

                index_ += 1

        # from event-based numpy array to event-based pandas data frame
        data_set_df = pandas.DataFrame(data=data_set[0:, 0:],
                                       index=[i for i in range(data_set.shape[0])],
                                       columns=['f' + str(i) for i in range(data_set.shape[1])])

        # reset data structure for encoding
        self.data_structure['encoding']['event_ids'] = {}
        self.data_structure['encoding']['context_attributes'] = []

        # encoding of columns
        data_set_df = self.encode_eventlog(args, eventlog_df)

        # update of data structure
        # num_values_context will be automatically set based on encode_eventlog
        self.set_number_control_flow_attributes()
        self.set_number_values_features()

        # reset of data structure for get_sequence_from_encoded_eventlog
        self.data_structure['data']['ids_process_instances'] = []
        self.data_structure['data']['process_instances'] = []
        self.data_structure['data']['context_attributes_process_instances'] = []

        # from event-based pandas data frame to instance-based
        self.get_sequences_from_encoded_eventlog(data_set_df)

        # update of data structure for get_2d_data_tensor
        self.set_max_length_process_instance()

        # from sequence to tensor
        data_set = self.get_2d_data_tensor()

        return data_set, label
