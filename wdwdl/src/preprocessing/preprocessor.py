from __future__ import division
import numpy
import pandas
import category_encoders
import copy
import wdwdl.src.utils.plot as plot
import wdwdl.src.utils.general as general
import tensorflow as tf
import pickle
import wdwdl.src.workarounds.workarounds as wa
import wdwdl.src.preprocessing.utils as utils
import wdwdl.src.trainer as trainer


class Preprocessor(object):

    data_structure = {
        'support': {
            'data_dir': "",
            'encoded_data_dir': "",
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
            'num_process_instances': 0,
            'flag_pred_split': True
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
            },
            'predict': {
                'process_instances': [],
                'context_attributes': [],
                'event_ids': []
            }
        }
    }

    def __init__(self, args, ex_id):
        """
        Creates a pre-processor object.
        :param args: pre-processor.
        """

        if ex_id > 0:
            self.reset_structure()

        general.llprint("Initialization ... \n")
        self.data_structure['support']['data_dir'] = args.data_dir + args.data_set
        self.data_structure['support']['encoded_data_dir'] = r'%s' % args.data_dir + r'encoded_%s' % args.data_set

        eventlog_df = pandas.read_csv(self.data_structure['support']['data_dir'], sep=';')

        self.data_structure['encoding']['eventlog_df'] = eventlog_df

        eventlog_df = self.encode_eventlog(args, eventlog_df, "init")
        self.set_number_control_flow_attributes()

        self.get_sequences_from_encoded_eventlog(args, eventlog_df)

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


    def reset_structure(self):
        self.data_structure = {
            'support': {
                'data_dir': "",
                'encoded_data_dir': "",
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
                'num_process_instances': 0,
                'flag_pred_split': True
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
                },
                'predict': {
                    'process_instances': [],
                    'context_attributes': [],
                    'event_ids': []
                }
            }
        }


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
            max([len(x) for x in self.data_structure['data']['process_instances']]),
            max([len(x) for x in self.data_structure['data']['predict']['process_instances']])
        )

    def encode_eventlog(self, args, eventlog_df, logic):

        # case
        encoded_eventlog_df = pandas.DataFrame(eventlog_df.iloc[:, 0])

        for column_name in eventlog_df:
            column_index = eventlog_df.columns.get_loc(column_name)

            # skip case and timestamp
            if column_index == 1 or column_index > 2:

                column = eventlog_df[column_name]
                column_data_type = utils.get_attribute_data_type(column)  # cat or num

                if column_index == 1:  # activity
                    if column_data_type == 'num':

                        column_with_end_mark = self.add_end_mark_to_event_column(column_name)[column_name]
                        self.save_mapping_of_encoded_events(column_with_end_mark, column_with_end_mark)
                        encoded_column = column

                    elif column_data_type == 'cat':
                        encoded_column = self.encode_column(args, 'event', column_name, column_data_type, logic)

                    if isinstance(encoded_column, pandas.DataFrame):
                        self.set_length_of_event_encoding(len(encoded_column.columns))
                    elif isinstance(encoded_column, pandas.Series):
                        self.set_length_of_event_encoding(1)

                else:  # context attribute
                    encoded_column = self.encode_column(args, 'context', column_name, column_data_type, logic)

                encoded_eventlog_df = encoded_eventlog_df.join(encoded_column)

            else:
                encoded_eventlog_df[column_name] = eventlog_df[column_name]

        encoded_eventlog_df.to_csv(self.data_structure['support']['encoded_data_dir'], sep=';', index=False)

        return encoded_eventlog_df

    def encode_column(self, args, attribute_type, attribute_name, column_data_type, logic):

        mode = utils.get_encoding_mode(args, column_data_type)

        if mode == 'min_max_norm':
            encoded_column = self.apply_min_max_normalization(attribute_name)

        elif mode == 'bin':
            if logic == "init":
                encoded_column = self.create_encoder(attribute_type, attribute_name, mode)
            else:
                encoded_column = self.load_encoder(attribute_type, attribute_name)

        else:
            # no encoding
            encoded_column = self.data_structure['encoding']['eventlog_df'][attribute_name]

        return encoded_column


    def apply_min_max_normalization(self, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        data = dataframe[column_name].fillna(dataframe[column_name].mean())
        encoded_data = data / data.max()

        self.set_length_of_context_encoding(1)

        return encoded_data

    def load_encoder(self, attribute_type, column_name):
        dataframe = self.data_structure['encoding']['eventlog_df']

        if attribute_type == 'event':
            dataframe = self.add_end_mark_to_event_column(column_name)

        # load encoder
        pkl_file = open('./encoder/%s.pkl' % column_name, 'rb')
        encoder = pickle.load(pkl_file)
        encoded_df = encoder.transform(dataframe)

        encoded_data = encoded_df[
            encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        if attribute_type == 'event':
            self.save_mapping_of_encoded_events(dataframe[column_name], encoded_data)
            encoded_data = utils.remove_end_mark_from_event_column(encoded_data)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        return encoded_data


    def create_encoder(self, attribute_type, column_name, mode):
        dataframe = self.data_structure['encoding']['eventlog_df']

        if attribute_type == 'event':
            dataframe = self.add_end_mark_to_event_column(column_name)

        encoder = category_encoders.BinaryEncoder(cols=[column_name])
        encoded_df = encoder.fit_transform(dataframe)

        # save encoder
        output = open('encoder/%s.pkl' % column_name, 'wb')
        pickle.dump(encoder, output)
        output.close()

        encoded_data = encoded_df[
            encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        if attribute_type == 'event':
            self.save_mapping_of_encoded_events(dataframe[column_name], encoded_data)
            encoded_data = utils.remove_end_mark_from_event_column(encoded_data)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        return encoded_data


    def add_end_mark_to_event_column(self, column_name):

        data_frame = self.data_structure['encoding']['eventlog_df']
        end_mark = self.data_structure['support']['end_process_instance']

        df_columns = data_frame.columns
        new_row = []
        for column in df_columns:
            if column == column_name:
                new_row.append(end_mark)
            else:
                new_row.append(0)

        row_df = pandas.DataFrame([new_row], columns=data_frame.columns)
        data_frame = data_frame.append(row_df, ignore_index=True)

        return data_frame

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

    def get_sequences_from_raw_event_log(self, event_log_df):
        id_latest_process_instance_raw = ''
        process_instance_raw = ''
        first_event_of_process_instance_raw = True
        context_attributes_process_instance_raw = []

        for index, event in event_log_df.iterrows():

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
                context_attributes_event = utils.get_context_attributes_of_event(event)
                context_attributes_process_instance_raw.append(context_attributes_event)

            process_instance_raw.append(event[1])
            first_event_of_process_instance_raw = False

        self.add_data_to_data_structure(process_instance_raw, 'process_instances_raw')

        if self.data_structure['meta']['num_attributes_context'] > 0:
            self.add_data_to_data_structure(context_attributes_process_instance_raw,
                                            'context_attributes_process_instances_raw')

    def get_sequences_from_encoded_eventlog(self, args, eventlog_df):

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


        # add last process instance to data structure
        self.add_data_to_data_structure(process_instance, 'process_instances')
        # add last context attributes to data structure
        if self.data_structure['meta']['num_attributes_context'] > 0:
            self.add_data_to_data_structure(context_attributes_process_instance, 'context_attributes_process_instances')


        # first, filter out process instances for prediction
        if self.data_structure["meta"]["flag_pred_split"]:

            # get ides for data and prediction set
            ids_data_set, ids_pred_set, _, _ = general.train_test_sets_from_data_set(
                args,
                self.data_structure["data"]["ids_process_instances"],
                self.data_structure["data"]["ids_process_instances"],
                args.pred_split)

            """
            # get data for pred set
            process_instances_ = []
            context_attributes_ = []
            for id in ids_pred_set:
                process_instances_.append(self.data_structure["data"]["process_instances"][id])
                context_attributes_.append(self.data_structure["data"]["context_attributes_process_instances"][id])

            self.data_structure["data"]["predict"]["process_instances"] = process_instances_
            self.data_structure["data"]["predict"]["context_attributes"] = context_attributes_
            self.data_structure["data"]["predict"]["event_ids"] = ids_pred_set

            # remove prediction set from data set
            process_instances_ = []
            context_attributes_ = []
            for id in ids_data_set:
                process_instances_.append(self.data_structure["data"]["process_instances"][id])
                context_attributes_.append(self.data_structure["data"]["context_attributes_process_instances"][id])

            self.data_structure["data"]["process_instances"] = process_instances_
            self.data_structure["data"]["context_attributes_process_instances"] = context_attributes_
            self.data_structure["data"]["ids_process_instances"] = ids_data_set
            self.data_structure["meta"]["flag_pred_split"] = False
            """

            self.data_structure["data"]["predict"]["process_instances"] = [self.data_structure["data"]["process_instances"][index] for index in range(len(self.data_structure["data"]["process_instances"])) if index in ids_pred_set]
            self.data_structure["data"]["predict"]["context_attributes"] = [self.data_structure["data"]["context_attributes_process_instances"][index] for index in range(len(self.data_structure["data"]["context_attributes_process_instances"])) if index in ids_pred_set]
            self.data_structure["data"]["predict"]["event_ids"] = sorted(ids_pred_set)

            # remove prediction set from data set
            self.data_structure["data"]["process_instances"] = [self.data_structure["data"]["process_instances"][index] for index in range(len(self.data_structure["data"]["process_instances"])) if index in ids_data_set]
            self.data_structure["data"]["context_attributes_process_instances"] = [self.data_structure["data"]["context_attributes_process_instances"][index] for index in range(len(self.data_structure["data"]["context_attributes_process_instances"])) if index in ids_data_set]
            self.data_structure["data"]["ids_process_instances"] = sorted(ids_data_set)
            self.data_structure["meta"]["flag_pred_split"] = False



    def check_for_context_attributes_df(self, event):

        if len(event) == self.data_structure['encoding']['num_values_control_flow']:
            general.llprint("No context attributes found ...\n")
        else:
            self.data_structure['meta']['num_attributes_context'] = len(self.data_structure['encoding']['context_attributes'])
            self.data_structure['encoding']['num_values_context'] = sum(self.data_structure['encoding']['context_attributes'])
            general.llprint("%d context attributes found ...\n" % self.data_structure['meta']['num_attributes_context'])

    def add_encoded_event_to_process_instance(self, event, process_instance):

        encoded_event_id = []
        start_index = 1
        end_index = self.data_structure['encoding']['event_ids']['length'] + 1

        for enc_val in range(start_index, end_index):
            encoded_event_id.append(event[enc_val])

        process_instance.append(tuple(encoded_event_id))

        return process_instance

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

    def get_2d_data_tensor_prediction(self):
        process_instances = self.data_structure['data']['predict']['process_instances']
        context_attributes_process_instances = self.data_structure['data']['predict']['context_attributes']
        number_context_attributes = self.data_structure['encoding']['num_values_context']
        number_attributes = self.data_structure['encoding']['num_values_features'] - 2  # case + time
        vector_length = self.data_structure['meta']['max_length_process_instance'] * number_attributes

        # Create structure
        data_set = numpy.zeros((
            len(process_instances),
            vector_length
        ))

        # Fill data
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
                            number_context_attributes + number_event_attributes) + number_event_attributes + index_attribute] = \
                        context_attributes[index_attribute]

        return data_set

    def get_2d_data_tensor(self):

        process_instances = self.data_structure['data']['process_instances']
        context_attributes_process_instances = self.data_structure['data']['context_attributes_process_instances']
        number_context_attributes = self.data_structure['encoding']['num_values_context']
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
                                number_context_attributes + number_event_attributes) + number_event_attributes + index_attribute] = \
                        context_attributes[index_attribute]

        return data_set

    def clean_event_log(self, args, preprocessor):
        """
        clean the event log with an Autoencoder.
        :param preprocessor:
        :param args:
        :return:
        """

        general.llprint("Create data set as tensor ... \n")
        features_data = self.get_2d_data_tensor()

        general.llprint("Learn autoencoder model ... \n")
        best_ae_id = trainer.train_ae_noise_removing(args, features_data, preprocessor)
        autoencoder = self.load_autoencoder(args, best_ae_id)

        # Remove noise of event log data
        predictions = autoencoder.predict(features_data)
        mse = numpy.mean(numpy.power(features_data - predictions, 2), axis=1)
        mse_df = pandas.DataFrame({'reconstruction_error': mse}, index=[i for i in range(features_data.shape[0])])

        # Threshold used for ECIS2020 Paper
        # threshold = df_error['reconstruction_error'].median() + (df_error['reconstruction_error'].std() *
        # args.remove_noise_factor)

        threshold = numpy.percentile(mse, args.remove_noise_threshold)

        print(threshold)

        if args.remove_noise:

            no_outliers = mse_df.index[mse_df['reconstruction_error'] <= threshold].tolist()

        else:
            no_outliers = mse_df.index[mse_df['reconstruction_error'] >= 0].tolist()

        general.llprint("Number of outliers: %i\n" % (len(features_data) - len(no_outliers)))
        general.llprint("Number of no outliers: %i\n" % len(no_outliers))
        plot.export_reconstruction_error(args, mse_df, 'reconstruction_error', threshold)

        if args.verbose:
            plot.reconstruction_error(args, mse_df, 'reconstruction_error', threshold)

        return no_outliers

    def load_autoencoder(self, args, best_ae_id):
        """
        Loads the previously build encoder model.
        :param args:
        :param best_ae_id:
        :return:
        """

        if best_ae_id == -1:
            autoencoder = tf.keras.models.load_model('%sae.h5' % args.checkpoint_dir)
        else:
            autoencoder = tf.keras.models.load_model('%sae_trial%s.h5' % (args.checkpoint_dir, best_ae_id))

        return autoencoder

    def add_workarounds_to_event_log(self, args, no_outliers):
        """
        Adds workarounds to instances of an event log.
        :param args:
        :param no_outliers:
        :return:
        """

        # Get process instances
        eventlog_df = self.data_structure['encoding']['eventlog_df']
        self.get_sequences_from_raw_event_log(eventlog_df)

        # Remove pred set from raw structures
        self.data_structure["data"]["process_instances_raw"] = [self.data_structure["data"]["process_instances_raw"][index] for index in range(len(self.data_structure["data"]["process_instances_raw"])) if index in self.data_structure["data"]["ids_process_instances"]]
        self.data_structure["data"]["context_attributes_process_instances_raw"] = [self.data_structure["data"]["context_attributes_process_instances_raw"][index] for index in range(len(self.data_structure["data"]["context_attributes_process_instances_raw"])) if index in self.data_structure["data"]["ids_process_instances"]]

        # Raw structures
        process_instances = self.data_structure['data']['process_instances_raw']
        process_instances_context = self.data_structure['data']['context_attributes_process_instances_raw']

        # Get process instance without noise
        process_instances_ = []
        process_instances_context_ = []
        num_process_instances = len(process_instances)
        ids_normal_process_instances = []

        # Normal data
        for index in range(0, num_process_instances):
            if index in no_outliers and len(process_instances[index]) >= 2:
                ids_normal_process_instances.append(index)
                process_instances_.append(process_instances[index])
                process_instances_context_.append(process_instances_context[index])

        # Augment data
        if args.remove_noise_augmentation:
            # Number of augmented process instances
            num_augmented_process_instances = num_process_instances - len(process_instances_)

            # Removed noisy data is replaced by normal data
            for index in range(0, num_augmented_process_instances):

                # Get random process instance
                index_ = numpy.random.choice(ids_normal_process_instances, 1, replace=True)

                process_instances_.append(process_instances[index_[0]])
                process_instances_context_.append(process_instances_context[index_[0]])

        # Add workaround
        process_instances_wa = []
        process_instances_context_wa = []
        label = [0] * len(process_instances_)  # 0 means that a process instance does not include a workaround
        probability = 0.3  # 30% of the process instances include workarounds
        unique_events = utils.get_unique_events(process_instances)
        unique_context = utils.get_unique_context(process_instances_context)

        for index in range(0, len(process_instances_)):

            if numpy.random.uniform(0, 1) <= probability:

                workaround_form = int(numpy.random.uniform(1, 7 + 1))  # + 1 since it excludes the upper bound

                if workaround_form == 1:  # injured_responsibility
                    process_instance_wa, process_instance_context_wa = \
                        wa.injured_responsibility(
                            process_instances_[index],
                            process_instances_context_[index],
                            unique_context,
                            max_injures=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 2:  # manipulated_data
                    process_instance_wa, process_instance_context_wa = \
                        wa.manipulated_data(
                            process_instances_[index],
                            process_instances_context_[index],
                            unique_context,
                            max_events=1,
                            max_attributes=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 3:  # repeated_activity
                    process_instance_wa, process_instance_context_wa = \
                        wa.repeated_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_repetitions=1,
                            max_repetition_length=10,
                            min_repetition_length=3
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 4:  # substituted_activity
                    process_instance_wa, process_instance_context_wa = \
                        wa.substituted_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            unique_events,
                            process_instances_,
                            process_instances_context_,
                            max_substitutions=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 5:  # interchanged_activity
                    process_instance_wa, process_instance_context_wa = \
                         wa.interchanged_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_interchanges=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 6:  # bypassed_activity
                    process_instance_wa, process_instance_context_wa = \
                        wa.bypassed_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            max_sequence_size=3
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

                elif workaround_form == 7:  # added_activity
                    process_instance_wa, process_instance_context_wa = \
                        wa.added_activity(
                            process_instances_[index],
                            process_instances_context_[index],
                            unique_events,
                            process_instances_,
                            process_instances_context_,
                            max_adds=1
                        )
                    process_instances_wa.append(process_instance_wa)
                    process_instances_context_wa.append(process_instance_context_wa)
                    label[index] = workaround_form

            else:
                process_instances_wa.append(process_instances_[index])
                process_instances_context_wa.append(process_instances_context_[index])

        # From instance-based list to event-based numpy array
        number_of_events = sum(list([len(element) for element in process_instances_wa]))
        data_set = numpy.zeros((number_of_events, 3 + len(process_instances_context_wa[0][0])))  # case, event and time

        index_ = 0
        for index_instance in range(0, len(process_instances_wa)):
            for index_event in range(0, len(process_instances_wa[index_instance])):
                # Case
                data_set[index_, 0] = index_instance

                # Event
                data_set[index_, 1] = process_instances_wa[index_instance][index_event]

                # Time
                data_set[index_, 2] = 0  # only for filling the gap

                # Context attributes
                for index_context in range(0, len(process_instances_context_wa[index_instance][index_event])):
                    data_set[index_, index_context + 3] = process_instances_context_wa[index_instance][index_event][
                        index_context]
                index_ += 1

        # From event-based numpy array to event-based pandas data frame
        data_set_df = pandas.DataFrame(data=data_set[0:, 0:],
                                       index=[i for i in range(data_set.shape[0])],
                                       columns=[i for i in eventlog_df.columns])
        for x in eventlog_df.columns:
            data_set_df[x] = data_set_df[x].astype(eventlog_df[x].dtypes.name)

        # Reset data structure for encoding
        self.data_structure['encoding']['event_ids'] = {}
        self.data_structure['encoding']['context_attributes'] = []
        self.data_structure['encoding']['eventlog_df'] = data_set_df

        # Encoding of columns
        data_set_df = self.encode_eventlog(args, data_set_df, "use")

        # Reset of data structure for get_sequence_from_encoded_eventlog
        self.data_structure['data']['ids_process_instances'] = []
        self.data_structure['data']['process_instances'] = []
        self.data_structure['data']['context_attributes_process_instances'] = []

        # From event-based pandas data frame to instance-based
        self.get_sequences_from_encoded_eventlog(args, data_set_df)

        # Update of data structure
        # Num_values_context will be automatically set based on encode_eventlog and length of event ids
        self.set_number_control_flow_attributes()
        self.set_number_values_features()

        # Update of data structure for get_2d_data_tensor
        self.set_max_length_process_instance()

        # From sequence to tensor
        data_set = self.get_2d_data_tensor()

        return data_set, label

    def prepare_event_log_for_prediction(self):
        general.llprint("Create data set as tensor ... \n")
        features_data = self.get_2d_data_tensor_prediction()

        return features_data


