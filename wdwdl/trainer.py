from __future__ import print_function, division
import keras_contrib
import keras
from datetime import datetime


def train_nn_wa_classification(args, data_set, label):
    """ We use an deep artificial neural network for learning the mapping of process instances and the label. """

    input_layer = keras.layers.Input(shape=(data_set.shape[1], ), name='input_layer')
    layer_1 = keras.layers.Dense(100, activation='tanh')(input_layer)
    layer_2 = keras.layers.Dropout(0.2)(layer_1)
    layer_2 = keras.layers.Dense(100, activation='tanh')(layer_2)
    layer_3 = keras.layers.Dropout(0.2)(layer_2)
    layer_3 = keras.layers.Dense(100, activation='tanh')(layer_3)
    b1 = keras.layers.Dropout(0.2)(layer_3)
    output = keras.layers.core.Dense(label.shape[1], activation='softmax', name='output',
                                           kernel_initializer='glorot_uniform')(b1)

    model = keras.models.Model(inputs=[input_layer], outputs=[output])
    optimizer = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                       schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint('%sclf_wa_mapping.h5' % args.checkpoint_dir,
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()
    # start_training_time = datetime.now()
    model.fit(data_set, {'output': label}, validation_split=1 / args.num_folds, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    # training_time = datetime.now() - start_training_time

    # return training_time.total_seconds()


def train(args, preprocessor):
    preprocessor.set_training_set()
    features_data = preprocessor.data_structure['data']['train']['features_data']
    labels = preprocessor.data_structure['data']['train']['labels']
    max_length_process_instance = preprocessor.data_structure['meta']['max_length_process_instance']
    # num_features = preprocessor.data_structure['meta']['num_features']
    num_features = preprocessor.data_structure['encoding']['num_values_features'] - 2  # time + case
    num_event_ids = preprocessor.data_structure['meta']['num_event_ids']

    print('Create machine learning model ... \n')

    # Vanilla LSTM
    if args.dnn_architecture == 0:
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        l1 = keras.layers.recurrent.LSTM(128, implementation=2, activation="tanh", kernel_initializer='glorot_uniform',
                                         return_sequences=False, dropout=0.2)(main_input)
        b1 = keras.layers.normalization.BatchNormalization()(l1)

    # Stacked LSTM
    elif args.dnn_architecture == 1:

        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        l1 = keras.layers.recurrent.LSTM(128, activation='tanh', kernel_initializer='glorot_uniform',
                                         return_sequences=True, dropout=0.2)(main_input)
        l2 = keras.layers.recurrent.LSTM(128, activation='tanh', kernel_initializer='glorot_uniform',
                                         return_sequences=False, dropout=0.2)(l1)
        b1 = keras.layers.normalization.BatchNormalization()(l2)



    # Embedding (word2vec) + ANN
    # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/    
    elif args.dnn_architecture == 6:
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        emb = keras.layers.embeddings.Embedding(input_dim=1000, output_dim=32)(main_input)
        l1 = keras.layers.Flatten()(emb)
        l2 = keras.layers.core.Dense(128, activation='tanh')(l1)
        l3 = keras.layers.core.Dropout(0.2)(l2)
        b1 = keras.layers.normalization.BatchNormalization()(l3)

    # Multi Layer Perceptron (MLP)
    elif args.dnn_architecture == 9:
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        # flatten/reshape because when multivariate all should be on the same axis 
        input_layer_flattened = keras.layers.Flatten()(main_input)
        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)
        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)
        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)
        b1 = keras.layers.Dropout(0.3)(layer_3)

    # Encoder
    elif args.dnn_architecture == 12:
        # how to install keras contrib
        # pip install git+https://www.github.com/keras-team/keras-contrib.git
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(main_input)
        conv1 = keras_contrib.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.core.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)
        conv2 = keras_contrib.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.core.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
        conv3 = keras_contrib.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.core.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.core.Dense(units=256, activation='sigmoid')(multiply_layer)
        dense_layer = keras_contrib.layers.InstanceNormalization()(dense_layer)
        # output layer
        b1 = keras.layers.Flatten()(dense_layer)


    event_output = keras.layers.core.Dense(num_event_ids, activation='softmax', name='event_output',
                                           kernel_initializer='glorot_uniform')(b1)
    model = keras.models.Model(inputs=[main_input], outputs=[event_output])

    optimizer = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                       schedule_decay=0.004, clipvalue=3)
    # optimizer = keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.004, clipvalue=3)
    # optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    # optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model.compile(loss={'event_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_%s.h5' % (args.checkpoint_dir,
                                                                          preprocessor.data_structure['support'][
                                                                              'iteration_cross_validation']),
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()

    start_training_time = datetime.now()

    model.fit(features_data, {'event_output': labels}, validation_split=1 / args.num_folds, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    return training_time.total_seconds()
